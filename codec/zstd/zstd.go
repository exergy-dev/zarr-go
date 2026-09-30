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
// Encoders and decoders from klauspost/compress/zstd are safe for concurrent
// use via EncodeAll / DecodeAll, so the single pair New builds is shared by
// every caller for the lifetime of the Codec.
type Codec struct {
	enc *kpzstd.Encoder
	dec *kpzstd.Decoder
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
// The decoder's max-memory cap is sampled from declimit.Max() here and fixed
// for the lifetime of the Codec (klauspost's decoder takes its
// max-decoded-size limit at construction; changing zarr.SetMaxDecodedBytes
// afterward has no effect on already-built codecs). Without it, DecodeAll's
// own default cap is 64 GiB, which still permits a large single allocation
// before declimit's post-hoc check in DecodeBB ever runs.
func New(raw json.RawMessage) (*Codec, error) {
	var cfg configJSON
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, fmt.Errorf("zstd codec: invalid configuration: %w", err)
		}
	}
	level := 0
	if cfg.Level != nil {
		level = *cfg.Level
	}
	checksum := cfg.Checksum != nil && *cfg.Checksum

	encOpts := []kpzstd.EOption{
		// Array.Write encodes chunks sequentially, so extra slots sit idle yet each preallocates match tables (up to ~34 MiB at level >= 10).
		kpzstd.WithEncoderConcurrency(min(4, runtime.GOMAXPROCS(0))),
		kpzstd.WithEncoderCRC(checksum),
		// Single-segment frames include Frame_Content_Size in the
		// header, which numcodecs/zstandard.decompress requires.
		kpzstd.WithSingleSegment(true),
	}
	if level != 0 {
		encOpts = append(encOpts, kpzstd.WithEncoderLevel(kpzstd.EncoderLevelFromZstd(level)))
	}
	enc, err := kpzstd.NewWriter(nil, encOpts...)
	if err != nil {
		return nil, fmt.Errorf("zstd codec: encoder init: %w", err)
	}
	// Concurrency 0 (GOMAXPROCS) so Array.Read's NumCPU workers sharing this Codec don't serialize.
	decOpts := []kpzstd.DOption{kpzstd.WithDecoderConcurrency(0)}
	if maxDec := declimit.Max(); maxDec > 0 {
		decOpts = append(decOpts, kpzstd.WithDecoderMaxMemory(uint64(maxDec)))
	}
	dec, err := kpzstd.NewReader(nil, decOpts...)
	if err != nil {
		_ = enc.Close()
		return nil, fmt.Errorf("zstd codec: decoder init: %w", err)
	}
	return &Codec{enc: enc, dec: dec}, nil
}

// Name returns the codec's v3 spec name.
func (c *Codec) Name() string { return "zstd" }

// EncodeBB compresses in and returns a freshly allocated buffer of zstd
// frame bytes.
func (c *Codec) EncodeBB(in []byte) ([]byte, error) {
	scratchP := scratchPool.Get().(*[]byte)
	*scratchP = c.enc.EncodeAll(in, (*scratchP)[:0])
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
