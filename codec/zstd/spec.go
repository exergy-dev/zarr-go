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
//
// The process shares one encoder per level, sized for GOMAXPROCS concurrent
// encodes, and each slot preallocates match tables: about 1 MiB at the
// default levels, about 34 MiB from level 10 up. On a 32-thread machine
// the first encode at level 19 therefore allocates about 1 GiB, once.
func Spec(level int) codec.Spec {
	return codec.Spec{Name: "zstd", Configuration: json.RawMessage(`{"level":` + strconv.Itoa(level) + `}`)}
}
