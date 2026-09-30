//go:build !cgo || !zarrcgo

// Package blosc provides a stub for the Zarr v3 "blosc" bytes→bytes codec
// in the default (no-cgo) build. The codec is registered so callers can
// always look it up, but Configure immediately fails with
// codec.ErrRequiresCGO so the array-open path can produce a clear error
// telling the user to rebuild with `-tags zarrcgo`.
//
// To get a working blosc codec, install libblosc and rebuild:
//
//	# Debian / Ubuntu
//	sudo apt install libblosc-dev
//	go build -tags zarrcgo ./...
package blosc

import (
	"encoding/json"

	"github.com/exergy-dev/zarr-go/codec"
)

// Codec is the no-op stub used when the binary was not built with cgo +
// the `zarrcgo` build tag. Configure always returns codec.ErrRequiresCGO.
type Codec struct{}

// Name returns the codec's v3 spec name.
func (c *Codec) Name() string { return "blosc" }

// Configure always returns codec.ErrRequiresCGO in this build.
func (c *Codec) Configure(_ json.RawMessage) error {
	return codec.ErrRequiresCGO
}

// EncodeBB always returns codec.ErrRequiresCGO; it should never be
// reached because Configure fails first.
func (c *Codec) EncodeBB(_ []byte) ([]byte, error) {
	return nil, codec.ErrRequiresCGO
}

// DecodeBB always returns codec.ErrRequiresCGO; it should never be
// reached because Configure fails first.
func (c *Codec) DecodeBB(_ []byte) ([]byte, error) {
	return nil, codec.ErrRequiresCGO
}
