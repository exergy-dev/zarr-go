//go:build !cgo || !zarrcgo

// Package blosc provides a stub for the Zarr v3 "blosc" bytes→bytes codec
// in the default (no-cgo) build. The codec is registered so callers can
// always look it up, but New always fails with codec.ErrRequiresCGO so the
// array-open path can produce a clear error telling the user to rebuild
// with `-tags zarrcgo`.
//
// To get a working blosc codec, install libblosc and rebuild:
//
//	# Debian / Ubuntu
//	sudo apt install libblosc-dev
//	go build -tags zarrcgo ./...
package blosc

import (
	"encoding/json"
	"fmt"

	"github.com/exergy-dev/zarr-go/codec"
)

// Codec is the stub used when the binary was not built with cgo + the
// `zarrcgo` build tag. New never returns one; its methods exist only so the
// type has the same API in every build.
type Codec struct{}

// New always returns an error wrapping codec.ErrRequiresCGO in this build.
func New(_ json.RawMessage) (*Codec, error) {
	return nil, fmt.Errorf("blosc codec: %w (rebuild with -tags zarrcgo and libblosc)", codec.ErrRequiresCGO)
}

// Name returns the codec's v3 spec name.
func (c *Codec) Name() string { return "blosc" }

// EncodeBB always returns codec.ErrRequiresCGO; it is unreachable because
// New never builds a stub Codec.
func (c *Codec) EncodeBB(_ []byte) ([]byte, error) {
	return nil, codec.ErrRequiresCGO
}

// DecodeBB always returns codec.ErrRequiresCGO; it is unreachable because
// New never builds a stub Codec.
func (c *Codec) DecodeBB(_ []byte) ([]byte, error) {
	return nil, codec.ErrRequiresCGO
}
