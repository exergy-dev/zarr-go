// Package zlib implements the Zarr v2 "zlib" bytes→bytes codec using the
// pure-Go compress/zlib stdlib package.
//
// v2 metadata refers to it as `{"id": "zlib", "level": N}`; the v2 translator
// emits a v3-shaped Spec `{name: "zlib", configuration: {"level": N}}` against
// which this codec is registered.
package zlib

import (
	"bytes"
	stdzlib "compress/zlib"
	"encoding/json"
	"fmt"
	"io"
	"sync"

	"github.com/exergy-dev/zarr-go/codec"
	"github.com/exergy-dev/zarr-go/internal/declimit"
)

// Codec is the "zlib" bytes→bytes codec. Construct it with New; its
// configuration is fixed at construction and it is safe for concurrent use.
type Codec struct {
	// level is the compression level: -1 (default) or 0..9.
	level int
}

type configJSON struct {
	Level *int `json:"level,omitempty"`
}

// New builds a zlib codec from its configuration object.
//
// Accepted shape: {"level": <int>} where <int> is in [0,9] or -1 for the
// zlib default. If the field is absent the zlib default is used.
func New(raw json.RawMessage) (*Codec, error) {
	c := &Codec{level: stdzlib.DefaultCompression}
	if len(raw) == 0 {
		return c, nil
	}
	var cfg configJSON
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("zlib codec: invalid configuration: %w", err)
	}
	if cfg.Level != nil {
		l := *cfg.Level
		if l == -1 {
			c.level = stdzlib.DefaultCompression
		} else if l < 0 || l > 9 {
			return nil, fmt.Errorf("zlib codec: level must be 0-9 or -1, got %d", l)
		} else {
			c.level = l
		}
	}
	return c, nil
}

// Name returns the codec's spec name.
func (c *Codec) Name() string { return "zlib" }

// Pools, each kept because benchmarks show it pays for itself (same scheme
// as the gzip codec):
//
//   - writerPools: a writer allocates ~800 KiB of compressor state; reusing
//     it cuts EncodeBB's allocations by ~95% on 64 KiB chunks. Reset cannot
//     change a writer's level, so there is one pool per level (index
//     level+1, covering -1..9).
//   - readerPool: reuses a reader's inflate state across DecodeBB calls.
//   - encScratch: the compressed size isn't known up front, so compressing
//     into reused scratch and copying out an exact-size result allocates
//     less than growing a fresh buffer. Buffers over 1 MiB are dropped
//     rather than pinned.
var (
	writerPools [11]sync.Pool
	readerPool  sync.Pool
	encScratch  = sync.Pool{New: func() any { return new(bytes.Buffer) }}
)

func (c *Codec) EncodeBB(in []byte) ([]byte, error) {
	buf := encScratch.Get().(*bytes.Buffer)
	buf.Reset()
	defer func() {
		if buf.Cap() <= 1<<20 {
			encScratch.Put(buf)
		}
	}()
	var pool *sync.Pool
	var w *stdzlib.Writer
	if c.level >= -1 && c.level <= 9 {
		pool = &writerPools[c.level+1]
		w, _ = pool.Get().(*stdzlib.Writer)
	}
	if w != nil {
		w.Reset(buf)
	} else {
		var err error
		if w, err = stdzlib.NewWriterLevel(buf, c.level); err != nil {
			return nil, fmt.Errorf("zlib codec: writer init: %w", err)
		}
	}
	if _, err := w.Write(in); err != nil {
		return nil, fmt.Errorf("zlib codec: write: %w", err)
	}
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("zlib codec: close: %w", err)
	}
	if pool != nil {
		pool.Put(w)
	}
	return bytes.Clone(buf.Bytes()), nil
}

func (c *Codec) DecodeBB(in []byte) ([]byte, error) {
	br := bytes.NewReader(in)
	zr, _ := readerPool.Get().(io.ReadCloser)
	var err error
	if zr != nil {
		err = zr.(stdzlib.Resetter).Reset(br, nil)
	} else {
		zr, err = stdzlib.NewReader(br)
	}
	if err != nil {
		return nil, fmt.Errorf("zlib codec: reader init: %w", err)
	}
	// Reset reinitializes all reader state, so the reader is reusable
	// whatever happens below. Unbinding br keeps the pool from pinning in.
	defer func() {
		br.Reset(nil)
		readerPool.Put(zr)
	}()
	maxDec := declimit.Max()
	var r io.Reader = zr
	if maxDec > 0 {
		// Reading at most maxDec+1 bytes tells "at the cap" from "over it"
		// without materializing the whole bomb.
		r = io.LimitReader(zr, maxDec+1)
	}
	// Unlike gzip, a zlib stream records no decoded size to presize from.
	out, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("zlib codec: read: %w", err)
	}
	if maxDec > 0 && int64(len(out)) > maxDec {
		return nil, fmt.Errorf("zlib codec: decompressed output exceeds %d bytes", maxDec)
	}
	return out, nil
}

func init() {
	codec.Register("zlib", func(raw json.RawMessage) (codec.Codec, error) {
		c, err := New(raw)
		if err != nil {
			return nil, err // not a typed-nil codec.Codec
		}
		return c, nil
	})
}
