package bytes

import (
	"encoding/json"

	"github.com/exergy-dev/zarr-go/codec"
)

// Spec returns the metadata entry for a bytes codec with the given byte
// order, for zarr.WithCodecs. zarr.WithCodecs adds a little-endian bytes
// codec itself when the list has no array→bytes codec, so Spec is only
// needed to select big-endian storage.
func Spec(e Endian) codec.Spec {
	endian := "little"
	if e == Big {
		endian = "big"
	}
	return codec.Spec{Name: "bytes", Configuration: json.RawMessage(`{"endian":"` + endian + `"}`)}
}
