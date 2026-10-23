package gzip

import (
	"encoding/json"
	"strconv"

	"github.com/exergy-dev/zarr-go/codec"
)

// Spec returns the metadata entry for a gzip codec at the given compression
// level (0-9), for zarr.WithCodecs.
func Spec(level int) codec.Spec {
	return codec.Spec{Name: "gzip", Configuration: json.RawMessage(`{"level":` + strconv.Itoa(level) + `}`)}
}
