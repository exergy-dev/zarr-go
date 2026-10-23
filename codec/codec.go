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

// ErrUnknownCodec is returned by BuildPipeline when metadata references a
// codec name that isn't in the registry. The top-level zarr package
// re-exports it as ErrUnknownCodec so callers can errors.Is against either
// form.
var ErrUnknownCodec = errors.New("codec: unknown codec")

// ArrayChunk is the in-memory representation of one chunk during decode/encode.
// Data is the typed element buffer in row-major order; its byte length must
// equal product(Shape) * Dtype.Size().
//
// Fill, when non-nil, is one element's worth of native-byte-order bytes
// representing the array's fill value. Codecs that need fill-aware
// semantics (notably sharding_indexed, which represents missing inner
// chunks) should consult it. Codecs that don't care MAY ignore it. Length,
// when non-zero, must equal the dtype's element size.
type ArrayChunk struct {
	Shape []int64
	Dtype string // canonical v3 dtype name (e.g., "float32")
	Data  []byte
	Fill  []byte
}

// Codec is the common interface; every codec is one of the three kinds below.
//
// Codecs must be safe for concurrent use; a Pipeline is shared by all of an
// Array's reader goroutines. A codec is fully configured by the Factory that
// builds it and its configuration never changes afterwards.
type Codec interface {
	// Name returns the codec's v3 spec name.
	Name() string
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

// RangeFetcher is the minimal interface the sharding codec needs to read a
// shard from the underlying store: GetRange for inner chunks (and a
// leading index), GetSuffix for a trailing index. Both follow the
// store.Store contracts, and every store.Store satisfies it; it is declared
// here (not in store/) so codec packages don't have to import store.
type RangeFetcher interface {
	GetRange(ctx context.Context, key string, off, n int64) ([]byte, error)
	GetSuffix(ctx context.Context, key string, n int64) ([]byte, error)
}

// DimRange describes a half-open range [Start, Stop) along one dimension,
// with a non-negative Step; Step 0 is equivalent to Step 1. It is the one
// range type shared by the zarr package's selections, the partial-decode
// API, and the internal copy kernel.
//
// Within a PartialDecoder, the region is expressed in chunk-local coordinates
// (i.e., relative to the outer chunk's origin, not the array origin) and the
// produced output is the dense, row-major buffer of the selected sub-region
// with Step==1 along every dim. Stepped selections must be expanded by the
// caller before invoking PartialDecoder.
type DimRange struct {
	Start, Stop, Step int64
}

// Length returns the number of elements selected along this dim.
func (r DimRange) Length() int64 {
	if r.Step < 0 || r.Stop <= r.Start {
		return 0
	}
	if r.Step == 0 {
		return r.Stop - r.Start
	}
	return (r.Stop - r.Start + r.Step - 1) / r.Step
}

// PartialReadRequest bundles the arguments to PartialDecoder.DecodePartial.
type PartialReadRequest struct {
	// Key identifies the outer chunk blob in the store.
	Key string
	// FullShape is the outer chunk's FULL stored shape — the configured
	// chunk_shape, never clipped to array bounds. Chunks (including shards)
	// are always encoded at the full configured chunk_shape; clipping to
	// array bounds is a read-time concern the engine handles separately by
	// bounding Region. Passing a clipped shape here corrupts the inner
	// chunk grid computation (wrong index size, garbage entries).
	FullShape []int64
	// Dtype is the array's canonical v3 dtype name.
	Dtype string
	// Region is the chunk-local sub-region to decode (Step must be 1 along
	// every dim).
	Region []DimRange
	// Out is the pre-sized output buffer: product(region lengths) *
	// elementSize(Dtype) bytes, populated in row-major order matching the
	// region shape.
	Out []byte
	// Fill, when non-nil, is one element's worth of native-byte-order bytes
	// representing the array's fill value. Sharded codecs use it to fill
	// regions corresponding to missing inner chunks; codecs that don't care
	// MAY ignore it.
	Fill []byte
	// GapBytes is the maximum byte gap between two inner-chunk ranges that
	// will still be coalesced into a single fetch. Negative values are
	// treated as 0 (no coalescing beyond exact adjacency).
	GapBytes int64
}

// PartialDecoder is implemented by ArrayToBytesCodec implementations that
// can decode a sub-region of an outer chunk without materializing the full
// chunk. The engine probes for it on the AB codec; if present and the read
// is partial (and there are no BB codecs on top of the AB codec), the engine
// calls DecodePartial in place of fetching the whole chunk + Decode.
//
// fetcher issues range reads against the underlying store; req.Key
// identifies the outer chunk blob.
type PartialDecoder interface {
	DecodePartial(ctx context.Context, fetcher RangeFetcher, req *PartialReadRequest) error
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

// Factory constructs a ready-to-use codec from its configuration JSON object
// (the value of the v3 "configuration" field; nil or empty when the metadata
// has none). It returns an error if the configuration is invalid.
// BuildPipeline calls it once per codec spec at array open.
type Factory func(config json.RawMessage) (Codec, error)

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

// DefaultArrayToBytes is the little-endian bytes codec, the array→bytes
// codec EnsureArrayToBytes inserts.
var DefaultArrayToBytes = Spec{Name: "bytes", Configuration: json.RawMessage(`{"endian":"little"}`)}

// EnsureArrayToBytes returns specs with DefaultArrayToBytes inserted, in
// front of the first bytes→bytes codec (or at the end), when none of them is
// an array→bytes codec, so a caller can list only the codecs it cares about
// (typically just a compressor). Each spec is instantiated to classify it;
// an unknown codec or an invalid configuration is returned as an error (the
// same one BuildPipeline would report).
func EnsureArrayToBytes(specs []Spec) ([]Spec, error) {
	at := len(specs)
	for i, s := range specs {
		c, err := newCodec(s)
		if err != nil {
			return nil, err
		}
		if _, ok := c.(ArrayToBytesCodec); ok {
			return specs, nil
		}
		if _, ok := c.(BytesToBytesCodec); ok && i < at {
			at = i
		}
	}
	out := make([]Spec, 0, len(specs)+1)
	out = append(out, specs[:at]...)
	out = append(out, DefaultArrayToBytes)
	return append(out, specs[at:]...), nil
}

// newCodec instantiates the codec described by s from the registry.
func newCodec(s Spec) (Codec, error) {
	f := Lookup(s.Name)
	if f == nil {
		return nil, fmt.Errorf("%w: %q", ErrUnknownCodec, s.Name)
	}
	c, err := f(s.Configuration)
	if err != nil {
		return nil, fmt.Errorf("configure codec %q: %w", s.Name, err)
	}
	if c == nil {
		return nil, fmt.Errorf("configure codec %q: factory returned no codec", s.Name)
	}
	return c, nil
}

// BuildPipeline constructs a Pipeline from an ordered list of codec specs as
// they appear in v3 metadata. Exactly one ArrayToBytesCodec must be present.
// A configuration error is wrapped with the codec's name.
func BuildPipeline(specs []Spec) (*Pipeline, error) {
	p := &Pipeline{}
	seenAB := false
	for _, s := range specs {
		c, err := newCodec(s)
		if err != nil {
			return nil, err
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
		next := ArrayChunk{Shape: cur.Shape, Dtype: cur.Dtype, Fill: cur.Fill}
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
		tmp := ArrayChunk{Shape: out.Shape, Dtype: out.Dtype, Fill: out.Fill}
		if err := p.AA[i].DecodeAA(*out, &tmp); err != nil {
			return fmt.Errorf("codec %q: %w", p.AA[i].Name(), err)
		}
		*out = tmp
	}
	return nil
}
