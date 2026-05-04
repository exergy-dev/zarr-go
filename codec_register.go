package zarr

// Blank-import standard codec packages so their init() registrations run when
// callers import this package. Datasets that reference unregistered codecs
// fail at array open with a clear error.
import (
	_ "github.com/rkm/zarr-go/codec/blosc"
	_ "github.com/rkm/zarr-go/codec/bytes"
	_ "github.com/rkm/zarr-go/codec/crc32c"
	_ "github.com/rkm/zarr-go/codec/gzip"
	_ "github.com/rkm/zarr-go/codec/sharding"
	_ "github.com/rkm/zarr-go/codec/transpose"
	_ "github.com/rkm/zarr-go/codec/zlib"
	_ "github.com/rkm/zarr-go/codec/zstd"
)
