//go:build !cgo || !zarrcgo

// Package zstd implements the Zarr v3 "zstd" bytes→bytes codec using the
// pure-Go github.com/klauspost/compress/zstd library.
//
// Build with `-tags zarrcgo` (and cgo enabled) to swap in the cgo-backed
// implementation in cgo_zstd.go, which links libzstd directly.
package zstd

import (
	"bytes"
	"fmt"
	"runtime"
	"sync"

	kpzstd "github.com/klauspost/compress/zstd"

	"github.com/exergy-dev/zarr-go/internal/declimit"
)

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

// Codec is the v3 "zstd" bytes→bytes codec.
//
// Encoders and decoders from klauspost/compress/zstd are safe for concurrent
// use via EncodeAll / DecodeAll, so a single instance per Codec is reused.
// The instances are constructed lazily on first use and cached for the
// lifetime of the Codec.
type Codec struct {
	// Level is the zstd compression level. 0 means "library default".
	Level int
	// Checksum, when true, makes the encoder include the 32-bit XXH64
	// trailer in each frame.
	Checksum bool

	once    sync.Once
	initErr error
	enc     *kpzstd.Encoder
	dec     *kpzstd.Decoder
}

// init constructs the encoder/decoder pair on first use.
//
// The decoder's max-memory cap is sampled from declimit.Max() here, at first
// use, and fixed for the lifetime of this Codec instance (klauspost's
// decoder allocates its max-decoded-size limit at construction; changing
// zarr.SetMaxDecodedBytes afterward has no effect on already-constructed
// decoders). Without this, DecodeAll's own default cap is 64 GiB, which
// still permits a large single allocation before declimit's post-hoc check
// in DecodeBB below ever runs.
func (c *Codec) init() {
	c.once.Do(func() {
		encOpts := []kpzstd.EOption{
			// Array.Write encodes chunks sequentially, so extra slots sit idle yet each preallocates match tables (up to ~34 MiB at level >= 10).
			kpzstd.WithEncoderConcurrency(min(4, runtime.GOMAXPROCS(0))),
			kpzstd.WithEncoderCRC(c.Checksum),
			// Single-segment frames include Frame_Content_Size in the
			// header, which numcodecs/zstandard.decompress requires.
			kpzstd.WithSingleSegment(true),
		}
		if c.Level != 0 {
			encOpts = append(encOpts, kpzstd.WithEncoderLevel(kpzstd.EncoderLevelFromZstd(c.Level)))
		}
		enc, err := kpzstd.NewWriter(nil, encOpts...)
		if err != nil {
			c.initErr = fmt.Errorf("zstd codec: encoder init: %w", err)
			return
		}
		// Concurrency 0 (GOMAXPROCS) so Array.Read's NumCPU workers sharing this Codec don't serialize.
		decOpts := []kpzstd.DOption{kpzstd.WithDecoderConcurrency(0)}
		if maxDec := declimit.Max(); maxDec > 0 {
			decOpts = append(decOpts, kpzstd.WithDecoderMaxMemory(uint64(maxDec)))
		}
		dec, err := kpzstd.NewReader(nil, decOpts...)
		if err != nil {
			_ = enc.Close()
			c.initErr = fmt.Errorf("zstd codec: decoder init: %w", err)
			return
		}
		c.enc = enc
		c.dec = dec
	})
}

// EncodeBB compresses in and returns a freshly allocated buffer of zstd
// frame bytes.
func (c *Codec) EncodeBB(in []byte) ([]byte, error) {
	c.init()
	if c.initErr != nil {
		return nil, c.initErr
	}
	scratchP := scratchPool.Get().(*[]byte)
	*scratchP = c.enc.EncodeAll(in, (*scratchP)[:0])
	out := bytes.Clone(*scratchP)
	putScratch(scratchP)
	return out, nil
}

// DecodeBB decompresses one or more concatenated zstd frames in `in` and
// returns the resulting bytes.
func (c *Codec) DecodeBB(in []byte) ([]byte, error) {
	c.init()
	if c.initErr != nil {
		return nil, c.initErr
	}
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
