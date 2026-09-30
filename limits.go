package zarr

import "github.com/exergy-dev/zarr-go/internal/declimit"

// SetMaxDecodedBytes sets the maximum number of bytes any single codec
// invocation (gzip, zlib, zstd, blosc, ...) may decompress a chunk or shard
// index trailer into. This guards against decompression bombs: a small
// compressed payload whose header declares a huge decoded size. Pass 0 to
// disable the cap. The default is 1 GiB. Safe to call concurrently; the new
// value applies to subsequent codec invocations, except that a zstd codec
// fixes its decoder memory limit when built, at array open (see zstd.New).
func SetMaxDecodedBytes(n int64) { declimit.SetMax(n) }

// MaxDecodedBytes returns the cap currently set by SetMaxDecodedBytes.
func MaxDecodedBytes() int64 { return declimit.Max() }
