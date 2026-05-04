// Package gzip implements the Zarr v3 "gzip" bytes→bytes codec.
//
// The codec wraps the standard library's compress/gzip implementation. The
// v3 spec configuration object accepts a single optional integer field
// "level" in the inclusive range 0..9, mapped onto compress/gzip's level
// constants (0 = NoCompression, 1 = BestSpeed, 9 = BestCompression). When
// the field is absent the default level (5) is used; level == -1 explicitly
// selects gzip.DefaultCompression.
//
// The package name shadows the stdlib "compress/gzip" package; this file
// imports it under the alias "stdgzip".
package gzip

import (
	"bytes"
	stdgzip "compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"sync"

	"github.com/rkm/zarr-go/codec"
)

// Codec is the v3 "gzip" bytes→bytes codec.
type Codec struct {
	// Level is the compression level passed to compress/gzip. It is set by
	// Configure and read by EncodeBB. Valid values are -1 (default) and 0..9.
	Level int
}

type configJSON struct {
	Level *int `json:"level,omitempty"`
}

// Name returns the codec's v3 spec name.
func (c *Codec) Name() string { return "gzip" }

// Configure parses the v3 codec configuration object.
//
// Accepted shape: {"level": <int>} where <int> is in [0,9] or -1 for the
// gzip default. If the field is absent the default level (5) is used.
func (c *Codec) Configure(raw json.RawMessage) error {
	const defaultLevel = 5
	c.Level = defaultLevel
	if len(raw) == 0 {
		return nil
	}
	var cfg configJSON
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return fmt.Errorf("gzip codec: invalid configuration: %w", err)
	}
	if cfg.Level == nil {
		return nil
	}
	lvl := *cfg.Level
	switch {
	case lvl == -1:
		c.Level = stdgzip.DefaultCompression
	case lvl >= 0 && lvl <= 9:
		c.Level = lvl
	default:
		return fmt.Errorf("gzip codec: invalid level %d (want -1 or 0..9)", lvl)
	}
	return nil
}

// writerPools holds one sync.Pool of *gzip.Writer per compression level.
// The level is part of the pooled writer's state (Reset cannot change it),
// so we must pool by level.
var writerPools sync.Map // map[int]*sync.Pool

// readerPool holds reusable *gzip.Reader values; readers are level-agnostic.
var readerPool = sync.Pool{}

// bufferPool holds reusable *bytes.Buffer instances used as scratch space
// inside Encode/Decode. The buffer's backing array is the only allocation
// the codec needs to do per call besides the (caller-owned) returned slice.
// Pre-grown to 16 KiB on first acquisition to amortize the cost of small
// chunks; larger inputs grow on demand and the buffer keeps its capacity
// when returned to the pool.
var bufferPool = sync.Pool{
	New: func() any {
		buf := new(bytes.Buffer)
		buf.Grow(16 * 1024)
		return buf
	},
}

// putBuffer returns a buffer to the pool. It resets length but keeps cap.
// Buffers that have grown beyond a sane bound are dropped on the floor to
// avoid pinning huge allocations indefinitely.
func putBuffer(buf *bytes.Buffer) {
	const maxKeep = 1 << 20 // 1 MiB
	if buf.Cap() > maxKeep {
		return
	}
	buf.Reset()
	bufferPool.Put(buf)
}

// readerScratchPool reuses *bytes.Reader instances used to wrap the input
// slice for the gzip Reader. Each Reset rebinds it to a different slice.
var readerScratchPool = sync.Pool{
	New: func() any { return bytes.NewReader(nil) },
}

// getWriter returns a *gzip.Writer at the requested level, writing to w.
func getWriter(w io.Writer, level int) (*stdgzip.Writer, error) {
	pAny, ok := writerPools.Load(level)
	if !ok {
		pAny, _ = writerPools.LoadOrStore(level, &sync.Pool{})
	}
	p := pAny.(*sync.Pool)
	if v := p.Get(); v != nil {
		gw := v.(*stdgzip.Writer)
		gw.Reset(w)
		return gw, nil
	}
	gw, err := stdgzip.NewWriterLevel(w, level)
	if err != nil {
		return nil, err
	}
	return gw, nil
}

// putWriter returns a writer to the per-level pool.
func putWriter(gw *stdgzip.Writer, level int) {
	if pAny, ok := writerPools.Load(level); ok {
		pAny.(*sync.Pool).Put(gw)
	}
}

// EncodeBB gzip-compresses in and returns a new buffer.
//
// The intermediate scratch buffer is sourced from bufferPool and the final
// result is copied into a fresh slice owned by the caller. This keeps the
// pooled buffer free for reuse on the next call.
func (c *Codec) EncodeBB(in []byte) ([]byte, error) {
	buf := bufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	// Pre-grow the buffer; gzip output is rarely larger than the input plus
	// a small header, so this avoids most reallocations on typical data.
	if grow := len(in)/2 + 64; grow > buf.Cap() {
		buf.Grow(grow - buf.Cap())
	}

	gw, err := getWriter(buf, c.Level)
	if err != nil {
		putBuffer(buf)
		return nil, fmt.Errorf("gzip codec: new writer: %w", err)
	}
	if _, err := gw.Write(in); err != nil {
		// Discard the writer on error; its internal state is suspect.
		putBuffer(buf)
		return nil, fmt.Errorf("gzip codec: write: %w", err)
	}
	if err := gw.Close(); err != nil {
		putBuffer(buf)
		return nil, fmt.Errorf("gzip codec: close: %w", err)
	}
	putWriter(gw, c.Level)

	// Copy out so the pooled buffer can be reused. The result is a fresh
	// caller-owned slice; we don't use bufpool here because the caller
	// doesn't return EncodeBB output to the pool, and bufpool's overhead
	// per Get/Put is comparable to a single make for these sizes.
	src := buf.Bytes()
	out := make([]byte, len(src))
	copy(out, src)
	putBuffer(buf)
	return out, nil
}

// DecodeBB gzip-decompresses in and returns a new buffer.
//
// The intermediate scratch buffer is sourced from bufferPool and the final
// result is copied into a fresh slice owned by the caller.
func (c *Codec) DecodeBB(in []byte) ([]byte, error) {
	br := readerScratchPool.Get().(*bytes.Reader)
	br.Reset(in)

	var gr *stdgzip.Reader
	if v := readerPool.Get(); v != nil {
		gr = v.(*stdgzip.Reader)
		if err := gr.Reset(br); err != nil {
			// Don't return the reader to the pool: Reset failure means it is
			// not in a clean state. Still return the bytes.Reader.
			br.Reset(nil)
			readerScratchPool.Put(br)
			return nil, fmt.Errorf("gzip codec: reset reader: %w", err)
		}
	} else {
		var err error
		gr, err = stdgzip.NewReader(br)
		if err != nil {
			br.Reset(nil)
			readerScratchPool.Put(br)
			return nil, fmt.Errorf("gzip codec: new reader: %w", err)
		}
	}

	out := bufferPool.Get().(*bytes.Buffer)
	out.Reset()
	if grow := len(in) * 2; grow > out.Cap() {
		out.Grow(grow - out.Cap())
	}
	if _, err := io.Copy(out, gr); err != nil {
		// Close best-effort; the reader is not safe to reuse.
		_ = gr.Close()
		putBuffer(out)
		br.Reset(nil)
		readerScratchPool.Put(br)
		return nil, fmt.Errorf("gzip codec: read: %w", err)
	}
	if err := gr.Close(); err != nil {
		putBuffer(out)
		br.Reset(nil)
		readerScratchPool.Put(br)
		return nil, fmt.Errorf("gzip codec: close reader: %w", err)
	}
	readerPool.Put(gr)

	// Copy into a caller-owned, right-sized slice so the scratch buffer
	// can return to the pool. A plain make is cheaper than bufpool here:
	// the caller doesn't return DecodeBB output to the pool, and bufpool
	// rounds up to a power of two which would over-allocate.
	src := out.Bytes()
	result := make([]byte, len(src))
	copy(result, src)
	putBuffer(out)
	br.Reset(nil)
	readerScratchPool.Put(br)
	return result, nil
}

func init() {
	codec.Register("gzip", func() codec.Codec { return &Codec{} })
}
