package crc32c

import "github.com/exergy-dev/zarr-go/codec"

// Spec returns the metadata entry for a crc32c codec, for zarr.WithCodecs.
// It belongs after the compression codecs, so the checksum covers the bytes
// as stored.
func Spec() codec.Spec {
	return codec.Spec{Name: "crc32c"}
}
