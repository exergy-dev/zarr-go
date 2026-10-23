// Package zstd implements the Zarr v3 "zstd" bytes→bytes codec using the
// pure-Go github.com/klauspost/compress/zstd library.
package zstd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"runtime"
	"sync"

	kpzstd "github.com/klauspost/compress/zstd"

	"github.com/exergy-dev/zarr-go/codec"
	"github.com/exergy-dev/zarr-go/internal/declimit"
)

func init() {
	codec.Register("zstd", func(raw json.RawMessage) (codec.Codec, error) {
		c, err := New(raw)
		if err != nil {
			return nil, err // not a typed-nil codec.Codec
		}
		return c, nil
	})
}

// scratchPool holds EncodeBB's scratch space. The compressed size isn't
// known up front, so encoding into reused scratch and copying out an
// exact-size result keeps EncodeBB at one right-sized allocation; with a nil
// dst, EncodeAll allocates 2.5-4x more (and 9 times per 1 MiB chunk).
// DecodeBB needs no scratch: DecodeAll sizes its output from the frame
// header.
var scratchPool = sync.Pool{
	New: func() any {
		// Start at 16 KiB; klauspost will grow on demand.
		b := make([]byte, 0, 16*1024)
		return &b
	},
}

// putScratch returns a scratch slice to the pool, capped to a sane upper
// bound so we don't pin huge buffers indefinitely.
func putScratch(p *[]byte) {
	const maxKeep = 1 << 20 // 1 MiB
	if cap(*p) > maxKeep {
		return
	}
	*p = (*p)[:0]
	scratchPool.Put(p)
}

// Codec is the v3 "zstd" bytes→bytes codec. Construct it with New; the zero
// value is not usable.
//
// Decoding uses a decoder owned by the Codec. Encoding uses an encoder
// shared by every Codec in the process with the same level and checksum
// setting (see encoder): a klauspost encoder preallocates its match tables
// per concurrency slot, so one per open array would multiply that memory by
// the number of arrays, for a resource that EncodeAll already shares safely
// between goroutines.
type Codec struct {
	level    int
	checksum bool
	dec      *kpzstd.Decoder
}

type configJSON struct {
	Level    *int  `json:"level,omitempty"`
	Checksum *bool `json:"checksum,omitempty"`
}

// New builds a zstd codec from its v3 configuration object.
//
// Accepted shape: {"level": <int>, "checksum": <bool>}. Both fields are
// optional. "level" defaults to 0 (library default). "checksum" defaults to
// false; when true, the encoder includes the 32-bit XXH64 trailer in each
// frame.
//
// The decoder's max-memory limit is sampled from declimit.Max() here and
// fixed for the lifetime of the Codec (klauspost's decoder takes it only at
// construction), so a later zarr.SetMaxDecodedBytes does not affect
// already-built codecs. Without it, DecodeAll's own default limit is 64 GiB,
// which permits a huge allocation before DecodeBB's post-hoc check runs.
func New(raw json.RawMessage) (*Codec, error) {
	var cfg configJSON
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, fmt.Errorf("zstd codec: invalid configuration: %w", err)
		}
	}
	c := &Codec{}
	if cfg.Level != nil {
		c.level = *cfg.Level
	}
	c.checksum = cfg.Checksum != nil && *cfg.Checksum

	// Concurrency 0 (GOMAXPROCS) so Array.Read's decode workers sharing this
	// Codec don't serialize.
	decOpts := []kpzstd.DOption{kpzstd.WithDecoderConcurrency(0)}
	if maxDec := declimit.Max(); maxDec > 0 {
		decOpts = append(decOpts, kpzstd.WithDecoderMaxMemory(uint64(maxDec)))
	}
	dec, err := kpzstd.NewReader(nil, decOpts...)
	if err != nil {
		return nil, fmt.Errorf("zstd codec: decoder init: %w", err)
	}
	c.dec = dec
	return c, nil
}

// encKey identifies a shared encoder: everything that varies between
// encoders.
type encKey struct {
	level    int
	checksum bool
}

var (
	encMu    sync.Mutex
	encoders = map[encKey]*kpzstd.Encoder{}
)

// encoder returns the process-wide encoder for c's level and checksum
// setting, building it on first use. It is sized for GOMAXPROCS concurrent
// EncodeAll calls, which is what a concurrent Array.Write issues; beyond
// that, callers queue for a slot.
func (c *Codec) encoder() (*kpzstd.Encoder, error) {
	key := encKey{level: c.level, checksum: c.checksum}
	encMu.Lock()
	defer encMu.Unlock()
	if enc, ok := encoders[key]; ok {
		return enc, nil
	}
	opts := []kpzstd.EOption{
		kpzstd.WithEncoderConcurrency(runtime.GOMAXPROCS(0)),
		kpzstd.WithEncoderCRC(c.checksum),
		// Single-segment frames include Frame_Content_Size in the
		// header, which numcodecs/zstandard.decompress requires.
		kpzstd.WithSingleSegment(true),
	}
	if c.level != 0 {
		opts = append(opts, kpzstd.WithEncoderLevel(kpzstd.EncoderLevelFromZstd(c.level)))
	}
	enc, err := kpzstd.NewWriter(nil, opts...)
	if err != nil {
		return nil, fmt.Errorf("zstd codec: encoder init: %w", err)
	}
	encoders[key] = enc
	return enc, nil
}

func (c *Codec) Name() string { return "zstd" }

// EncodeBB compresses in and returns a freshly allocated buffer of zstd
// frame bytes.
func (c *Codec) EncodeBB(in []byte) ([]byte, error) {
	enc, err := c.encoder()
	if err != nil {
		return nil, err
	}
	scratchP := scratchPool.Get().(*[]byte)
	*scratchP = enc.EncodeAll(in, (*scratchP)[:0])
	out := bytes.Clone(*scratchP)
	putScratch(scratchP)
	return out, nil
}

// DecodeBB decompresses one or more concatenated zstd frames in `in` and
// returns the resulting bytes.
func (c *Codec) DecodeBB(in []byte) ([]byte, error) {
	// With a nil dst, DecodeAll sizes its output from the frame header's
	// content size (bounded by WithDecoderMaxMemory), so this is normally a
	// single right-sized allocation.
	out, err := c.dec.DecodeAll(in, nil)
	if err != nil {
		return nil, fmt.Errorf("zstd codec: decode: %w", err)
	}
	if maxDec := declimit.Max(); maxDec > 0 && int64(len(out)) > maxDec {
		return nil, fmt.Errorf("zstd codec: decompressed output exceeds %d bytes", maxDec)
	}
	return out, nil
}
