package blosc

import (
	"encoding/json"

	"github.com/exergy-dev/zarr-go/codec"
)

// init registers the "blosc" codec. New is provided by either blosc.go
// (cgo + `zarrcgo` build tag) or blosc_nocgo.go (the default build, where it
// returns codec.ErrRequiresCGO).
func init() {
	codec.Register("blosc", func(raw json.RawMessage) (codec.Codec, error) {
		c, err := New(raw)
		if err != nil {
			return nil, err // not a typed-nil codec.Codec
		}
		return c, nil
	})
}
