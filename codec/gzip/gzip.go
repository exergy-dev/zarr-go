// Package gzip implements the Zarr v3 "gzip" bytes→bytes codec.
//
// The codec wraps the klauspost/compress gzip implementation, a drop-in
// replacement for the standard library's compress/gzip with significantly
// faster decompression. The v3 spec configuration object accepts a single
// optional integer field "level" in the inclusive range 0..9, mapped onto
// gzip's level constants (0 = NoCompression, 1 = BestSpeed,
// 9 = BestCompression). When the field is absent the default level (5) is
// used; level == -1 explicitly selects gzip.DefaultCompression.
//
// The package name shadows the gzip package; this file imports it under the
// alias "kpgzip".
package gzip

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sync"

	"github.com/exergy-dev/zarr-go/codec"
	"github.com/exergy-dev/zarr-go/internal/declimit"
	kpgzip "github.com/klauspost/compress/gzip"
)

// Codec is the v3 "gzip" bytes→bytes codec. Construct it with New; its
// configuration is fixed at construction and it is safe for concurrent use.
type Codec struct {
	// level is the compression level passed to compress/gzip: -1 (default)
	// or 0..9.
	level int
}

type configJSON struct {
	Level *int `json:"level,omitempty"`
}

// New builds a gzip codec from its v3 configuration object.
//
// Accepted shape: {"level": <int>} where <int> is in [0,9] or -1 for the
// gzip default. If the field is absent the default level (5) is used.
func New(raw json.RawMessage) (*Codec, error) {
	const defaultLevel = 5
	c := &Codec{level: defaultLevel}
	if len(raw) == 0 {
		return c, nil
	}
	var cfg configJSON
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("gzip codec: invalid configuration: %w", err)
	}
	if cfg.Level == nil {
		return c, nil
	}
	lvl := *cfg.Level
	switch {
	case lvl == -1:
		c.level = kpgzip.DefaultCompression
	case lvl >= 0 && lvl <= 9:
		c.level = lvl
	default:
		return nil, fmt.Errorf("gzip codec: invalid level %d (want -1 or 0..9)", lvl)
	}
	return c, nil
}

// Name returns the codec's v3 spec name.
func (c *Codec) Name() string { return "gzip" }

// Pools, each kept because benchmarks show it pays for itself:
//
//   - writerPools: a writer carries ~1 MiB of compressor state; without
//     reuse EncodeBB is up to 2.5x slower on 64 KiB chunks. Reset cannot
//     change a writer's level, so there is one pool per level (index
//     level+1, covering -1..9).
//   - readerPool: reusing a reader's inflate state saves ~40 KiB and ~85
//     allocations per DecodeBB (5% serial, 35% parallel on 64 KiB chunks).
//   - encScratch: the compressed size isn't known up front, so compressing
//     into reused scratch and copying out an exact-size result allocates
//     ~60% less than growing a fresh buffer. Buffers over 1 MiB are dropped
//     rather than pinned.
var (
	writerPools [11]sync.Pool
	readerPool  sync.Pool
	encScratch  = sync.Pool{New: func() any { return new(bytes.Buffer) }}
)

// EncodeBB gzip-compresses in and returns a new buffer.
func (c *Codec) EncodeBB(in []byte) ([]byte, error) {
	buf := encScratch.Get().(*bytes.Buffer)
	buf.Reset()
	defer func() {
		if buf.Cap() <= 1<<20 {
			encScratch.Put(buf)
		}
	}()
	var pool *sync.Pool
	var gw *kpgzip.Writer
	if c.level >= -1 && c.level <= 9 {
		pool = &writerPools[c.level+1]
		gw, _ = pool.Get().(*kpgzip.Writer)
	}
	if gw != nil {
		gw.Reset(buf)
	} else {
		var err error
		if gw, err = kpgzip.NewWriterLevel(buf, c.level); err != nil {
			return nil, fmt.Errorf("gzip codec: new writer: %w", err)
		}
	}
	if _, err := gw.Write(in); err != nil {
		return nil, fmt.Errorf("gzip codec: write: %w", err)
	}
	if err := gw.Close(); err != nil {
		return nil, fmt.Errorf("gzip codec: close: %w", err)
	}
	if pool != nil {
		pool.Put(gw)
	}
	return bytes.Clone(buf.Bytes()), nil
}

// DecodeBB gzip-decompresses in and returns a new buffer.
func (c *Codec) DecodeBB(in []byte) ([]byte, error) {
	br := bytes.NewReader(in)
	gr, _ := readerPool.Get().(*kpgzip.Reader)
	var err error
	if gr != nil {
		err = gr.Reset(br)
	} else {
		gr, err = kpgzip.NewReader(br)
	}
	if err != nil {
		return nil, fmt.Errorf("gzip codec: new reader: %w", err)
	}
	// Reset reinitializes all reader state, so the reader is reusable
	// whatever happens below. Unbinding br keeps the pool from pinning in.
	defer func() {
		br.Reset(nil)
		readerPool.Put(gr)
	}()
	maxDec := declimit.Max()
	var r io.Reader = gr
	if maxDec > 0 {
		// Reading at most maxDec+1 bytes tells "at the cap" from "over it"
		// without materializing the whole bomb.
		r = io.LimitReader(gr, maxDec+1)
	}
	var out bytes.Buffer
	// Size the output from the trailer so it is read in one allocation;
	// the +MinRead leaves ReadFrom room to see EOF without growing.
	out.Grow(sizeHint(in, maxDec) + bytes.MinRead)
	if _, err := out.ReadFrom(r); err != nil {
		return nil, fmt.Errorf("gzip codec: read: %w", err)
	}
	if maxDec > 0 && int64(out.Len()) > maxDec {
		return nil, fmt.Errorf("gzip codec: decompressed output exceeds %d bytes (call zarr.SetMaxDecodedBytes to raise)", maxDec)
	}
	return out.Bytes(), nil
}

// sizeHint returns the decoded size recorded in a gzip stream's trailer
// (ISIZE, the last member's length mod 2^32). It is untrusted input used
// only as a capacity hint, so it is clamped to what deflate could produce
// from in (at most 1032:1) and to the declimit cap.
func sizeHint(in []byte, maxDec int64) int {
	if len(in) < 4 {
		return 0
	}
	n := int64(binary.LittleEndian.Uint32(in[len(in)-4:]))
	n = min(n, int64(len(in))*1032, math.MaxInt-bytes.MinRead)
	if maxDec > 0 {
		n = min(n, maxDec)
	}
	return int(n)
}

func init() {
	codec.Register("gzip", func(raw json.RawMessage) (codec.Codec, error) {
		c, err := New(raw)
		if err != nil {
			return nil, err // not a typed-nil codec.Codec
		}
		return c, nil
	})
}
