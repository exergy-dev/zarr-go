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

type Codec struct {
	Level int
}

type configJSON struct {
	Level *int `json:"level,omitempty"`
}

func (c *Codec) Name() string { return "zlib" }

func (c *Codec) Configure(raw json.RawMessage) error {
	c.Level = stdzlib.DefaultCompression
	if len(raw) == 0 {
		return nil
	}
	var cfg configJSON
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return fmt.Errorf("zlib codec: invalid configuration: %w", err)
	}
	if cfg.Level != nil {
		l := *cfg.Level
		if l == -1 {
			c.Level = stdzlib.DefaultCompression
		} else if l < 0 || l > 9 {
			return fmt.Errorf("zlib codec: level must be 0-9 or -1, got %d", l)
		} else {
			c.Level = l
		}
	}
	return nil
}

var readerPool = sync.Pool{}

func (c *Codec) EncodeBB(in []byte) ([]byte, error) {
	var buf bytes.Buffer
	buf.Grow(len(in) / 2)
	w, err := stdzlib.NewWriterLevel(&buf, c.Level)
	if err != nil {
		return nil, fmt.Errorf("zlib codec: writer init: %w", err)
	}
	if _, err := w.Write(in); err != nil {
		return nil, fmt.Errorf("zlib codec: write: %w", err)
	}
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("zlib codec: close: %w", err)
	}
	return buf.Bytes(), nil
}

func (c *Codec) DecodeBB(in []byte) ([]byte, error) {
	var r io.ReadCloser
	if pooled, ok := readerPool.Get().(io.ReadCloser); ok {
		if reset, ok := pooled.(zlibResetter); ok {
			if err := reset.Reset(bytes.NewReader(in), nil); err == nil {
				r = pooled
			}
		}
	}
	if r == nil {
		var err error
		r, err = stdzlib.NewReader(bytes.NewReader(in))
		if err != nil {
			return nil, fmt.Errorf("zlib codec: reader init: %w", err)
		}
	}
	defer func() {
		_ = r.Close()
		readerPool.Put(r)
	}()
	maxDec := declimit.Max()
	var rdr io.Reader = r
	if maxDec > 0 {
		rdr = io.LimitReader(r, maxDec+1)
	}
	out, err := io.ReadAll(rdr)
	if err != nil {
		return nil, fmt.Errorf("zlib codec: read: %w", err)
	}
	if maxDec > 0 && int64(len(out)) > maxDec {
		return nil, fmt.Errorf("zlib codec: decompressed output exceeds %d bytes", maxDec)
	}
	return out, nil
}

type zlibResetter interface {
	Reset(r io.Reader, dict []byte) error
}

func init() {
	codec.Register("zlib", func() codec.Codec { return &Codec{} })
}
