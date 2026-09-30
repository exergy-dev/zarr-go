// Package zstd registration. The Codec type is defined in either zstd.go
// (pure-Go default build) or cgo_zstd.go (build tag `zarrcgo` with cgo).
// Both implementations expose the same public API and Name() == "zstd".
package zstd

import "github.com/exergy-dev/zarr-go/codec"

func init() {
	codec.Register("zstd", func() codec.Codec { return &Codec{} })
}
