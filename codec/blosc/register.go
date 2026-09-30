package blosc

import "github.com/exergy-dev/zarr-go/codec"

// init registers the "blosc" codec. The Codec implementation is provided
// by either blosc.go (cgo + `zarrcgo` build tag) or blosc_nocgo.go (the
// default build, where Configure returns codec.ErrRequiresCGO).
func init() {
	codec.Register("blosc", func() codec.Codec { return &Codec{} })
}
