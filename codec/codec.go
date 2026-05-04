// Package codec defines the codec interfaces and the global codec registry.
//
// A codec pipeline matches the Zarr v3 model: an ordered sequence of three
// codec kinds. Encode walks the pipeline left-to-right; decode walks it
// right-to-left.
//
//	array→array  (e.g., transpose)
//	array→bytes  (exactly one; e.g., the bytes codec, sharding)
//	bytes→bytes  (compression: gzip, zstd, blosc, ...)
package codec

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
)

// ErrRequiresCGO is returned by codec implementations whose backing C library
// is unavailable in the current build (i.e., the binary was not built with
// the `zarrcgo` build tag, or cgo was disabled). It is the canonical sentinel
// the array-open path checks; the top-level zarr package re-exports it as
// ErrCodecRequiresCGO.
var ErrRequiresCGO = errors.New("codec requires cgo build")

// ArrayChunk is the in-memory representation of one chunk during decode/encode.
// Data is the typed element buffer in row-major order; its byte length must
// equal product(Shape) * Dtype.Size().
type ArrayChunk struct {
	Shape []int64
	Dtype string // canonical v3 dtype name (e.g., "float32")
	Data  []byte
}

// Codec is the common interface; every codec is one of the three kinds below.
type Codec interface {
	Name() string
	// Configure parses the codec's configuration JSON object (the value of
	// the v3 "configuration" field). Called once at array open.
	Configure(json.RawMessage) error
}

// ArrayToArrayCodec transforms an array chunk into another array chunk.
type ArrayToArrayCodec interface {
	Codec
	EncodeAA(in ArrayChunk, out *ArrayChunk) error
	DecodeAA(in ArrayChunk, out *ArrayChunk) error
}

// ArrayToBytesCodec serializes an array chunk to a byte buffer (and back).
// Exactly one such codec must appear in every pipeline.
type ArrayToBytesCodec interface {
	Codec
	EncodeAB(in ArrayChunk) ([]byte, error)
	DecodeAB(in []byte, out *ArrayChunk) error
}

// BytesToBytesCodec transforms a byte buffer into another byte buffer
// (typically compression).
type BytesToBytesCodec interface {
	Codec
	EncodeBB(in []byte) ([]byte, error)
	DecodeBB(in []byte) ([]byte, error)
}

// RangeFetcher is the minimal interface the sharding codec needs to read
// inner chunks from the underlying store. It exists here (not in store/) so
// codec packages don't have to import store. The scheduler implements it.
type RangeFetcher interface {
	GetRange(ctx context.Context, key string, off, n int64) ([]byte, error)
}

// Spec describes one codec entry as it appears in v3 metadata.
type Spec struct {
	Name          string          `json:"name"`
	Configuration json.RawMessage `json:"configuration,omitempty"`
}

// Pipeline is the constructed, configured codec chain for one array.
//
// Decode order: BB (in reverse) -> AB.Decode -> AA (in reverse).
// Encode order: AA -> AB.Encode -> BB.
type Pipeline struct {
	AA []ArrayToArrayCodec
	AB ArrayToBytesCodec
	BB []BytesToBytesCodec
}

// --- Registry --------------------------------------------------------------

// Factory constructs a fresh, unconfigured codec instance.
type Factory func() Codec

var (
	regMu sync.RWMutex
	reg   = map[string]Factory{}
)

// Register adds a codec factory to the global registry. Intended to be called
// from package init() of codec implementations. Compile-time registration
// only — there is no runtime API to unregister or override.
func Register(name string, f Factory) {
	regMu.Lock()
	defer regMu.Unlock()
	if _, dup := reg[name]; dup {
		panic("codec: duplicate registration for " + name)
	}
	reg[name] = f
}

// Lookup returns the factory for name, or nil if unknown.
func Lookup(name string) Factory {
	regMu.RLock()
	defer regMu.RUnlock()
	return reg[name]
}

// BuildPipeline constructs and configures a Pipeline from an ordered list of
// codec specs as they appear in v3 metadata. Exactly one ArrayToBytesCodec
// must be present.
func BuildPipeline(specs []Spec) (*Pipeline, error) {
	p := &Pipeline{}
	seenAB := false
	for _, s := range specs {
		f := Lookup(s.Name)
		if f == nil {
			return nil, fmt.Errorf("codec %q not registered", s.Name)
		}
		c := f()
		if err := c.Configure(s.Configuration); err != nil {
			return nil, fmt.Errorf("configure codec %q: %w", s.Name, err)
		}
		switch v := c.(type) {
		case ArrayToBytesCodec:
			if seenAB {
				return nil, fmt.Errorf("multiple array→bytes codecs in pipeline")
			}
			p.AB = v
			seenAB = true
		case ArrayToArrayCodec:
			if seenAB {
				return nil, fmt.Errorf("array→array codec %q after array→bytes", s.Name)
			}
			p.AA = append(p.AA, v)
		case BytesToBytesCodec:
			if !seenAB {
				return nil, fmt.Errorf("bytes→bytes codec %q before array→bytes", s.Name)
			}
			p.BB = append(p.BB, v)
		default:
			return nil, fmt.Errorf("codec %q implements no codec interface", s.Name)
		}
	}
	if !seenAB {
		return nil, fmt.Errorf("pipeline missing array→bytes codec")
	}
	return p, nil
}

// Encode runs the pipeline forward over a typed chunk, returning the encoded
// byte payload to be written to the store.
func (p *Pipeline) Encode(in ArrayChunk) ([]byte, error) {
	cur := in
	for _, c := range p.AA {
		next := ArrayChunk{Shape: cur.Shape, Dtype: cur.Dtype}
		if err := c.EncodeAA(cur, &next); err != nil {
			return nil, fmt.Errorf("codec %q: %w", c.Name(), err)
		}
		cur = next
	}
	buf, err := p.AB.EncodeAB(cur)
	if err != nil {
		return nil, fmt.Errorf("codec %q: %w", p.AB.Name(), err)
	}
	for _, c := range p.BB {
		next, err := c.EncodeBB(buf)
		if err != nil {
			return nil, fmt.Errorf("codec %q: %w", c.Name(), err)
		}
		buf = next
	}
	return buf, nil
}

// Decode runs the pipeline in reverse over input bytes, writing into out.
// out.Shape and out.Dtype must be set; out.Data is allocated/grown as needed.
func (p *Pipeline) Decode(in []byte, out *ArrayChunk) error {
	buf := in
	// bytes->bytes in reverse order
	for i := len(p.BB) - 1; i >= 0; i-- {
		decoded, err := p.BB[i].DecodeBB(buf)
		if err != nil {
			return fmt.Errorf("codec %q: %w", p.BB[i].Name(), err)
		}
		buf = decoded
	}
	if err := p.AB.DecodeAB(buf, out); err != nil {
		return fmt.Errorf("codec %q: %w", p.AB.Name(), err)
	}
	// array->array in reverse
	for i := len(p.AA) - 1; i >= 0; i-- {
		tmp := ArrayChunk{Shape: out.Shape, Dtype: out.Dtype}
		if err := p.AA[i].DecodeAA(*out, &tmp); err != nil {
			return fmt.Errorf("codec %q: %w", p.AA[i].Name(), err)
		}
		*out = tmp
	}
	return nil
}
