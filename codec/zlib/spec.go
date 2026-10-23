package zlib

import (
	"encoding/json"
	"strconv"

	"github.com/exergy-dev/zarr-go/codec"
)

// Spec returns the metadata entry for a zlib codec at the given compression
// level (0-9, or -1 for the zlib default), for zarr.WithCodecs.
func Spec(level int) codec.Spec {
	return codec.Spec{Name: "zlib", Configuration: json.RawMessage(`{"level":` + strconv.Itoa(level) + `}`)}
}
