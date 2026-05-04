//go:build !cgo || !zarrcgo

// Package zstd implements the Zarr v3 "zstd" bytes→bytes codec using the
// pure-Go github.com/klauspost/compress/zstd library.
//
// Build with `-tags zarrcgo` (and cgo enabled) to swap in the cgo-backed
// implementation in cgo_zstd.go, which links libzstd directly.
package zstd

import (
	"encoding/json"
	"fmt"
	"sync"

	kpzstd "github.com/klauspost/compress/zstd"
)

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

type configJSON struct {
	Level    *int  `json:"level,omitempty"`
	Checksum *bool `json:"checksum,omitempty"`
}

// Name returns the codec's v3 spec name.
func (c *Codec) Name() string { return "zstd" }

// Configure parses the v3 codec configuration object.
//
// Accepted shape: {"level": <int>, "checksum": <bool>}. Both fields are
// optional. "level" defaults to 0 (library default). "checksum" defaults to
// false.
func (c *Codec) Configure(raw json.RawMessage) error {
	c.Level = 0
	c.Checksum = false
	if len(raw) == 0 {
		return nil
	}
	var cfg configJSON
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return fmt.Errorf("zstd codec: invalid configuration: %w", err)
	}
	if cfg.Level != nil {
		c.Level = *cfg.Level
	}
	if cfg.Checksum != nil {
		c.Checksum = *cfg.Checksum
	}
	return nil
}

// init constructs the encoder/decoder pair on first use.
func (c *Codec) init() {
	c.once.Do(func() {
		encOpts := []kpzstd.EOption{
			kpzstd.WithEncoderConcurrency(1),
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
		dec, err := kpzstd.NewReader(nil, kpzstd.WithDecoderConcurrency(1))
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
	return c.enc.EncodeAll(in, nil), nil
}

// DecodeBB decompresses one or more concatenated zstd frames in `in` and
// returns the resulting bytes.
func (c *Codec) DecodeBB(in []byte) ([]byte, error) {
	c.init()
	if c.initErr != nil {
		return nil, c.initErr
	}
	out, err := c.dec.DecodeAll(in, nil)
	if err != nil {
		return nil, fmt.Errorf("zstd codec: decode: %w", err)
	}
	return out, nil
}

