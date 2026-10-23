package zstd

import (
	"encoding/json"
	"strconv"

	"github.com/exergy-dev/zarr-go/codec"
)

// Spec returns the metadata entry for a zstd codec at the given compression
// level, for zarr.WithCodecs. Levels follow the zstd CLI (1 fastest, 19
// slowest; 0 selects the library default). Frames carry no checksum; pass a
// codec.Spec with {"checksum": true} in its Configuration for one.
func Spec(level int) codec.Spec {
	return codec.Spec{Name: "zstd", Configuration: json.RawMessage(`{"level":` + strconv.Itoa(level) + `}`)}
}
