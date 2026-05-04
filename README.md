# zarr-go

[![CI](https://github.com/exergy-dev/zarr-go/actions/workflows/ci.yml/badge.svg)](https://github.com/exergy-dev/zarr-go/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/rkm/zarr-go.svg)](https://pkg.go.dev/github.com/rkm/zarr-go)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A pure-Go implementation of the [Zarr](https://zarr.dev) chunked, compressed,
N-dimensional array format. Read/write for v3, read for v2, with pluggable
codecs and storage backends.

- **Typed arrays.** `Array[T]` is generic over the element type — no
  `interface{}` round-trips on the hot path.
- **Pluggable codecs.** `bytes`, `transpose`, `gzip`, `zlib`, `zstd`, `blosc`,
  `crc32c`, and `sharding_indexed` (with partial decode and range coalescing).
- **Pluggable stores.** Local filesystem (with optional `mmap` fast path),
  in-memory, HTTP, S3, GCS, Azure Blob, plus an LRU cache decorator.
- **Concurrent I/O.** Per-chunk fetch+decode goroutines, bounded by user
  options.
- **Safety by default.** Atomic creates, decompression-bomb cap, fill-aware
  partial decode, lossless metadata round-trip.

## Install

```bash
go get github.com/rkm/zarr-go
```

Requires Go 1.25 or newer.

## Quick start

### Read

```go
package main

import (
	"context"
	"fmt"

	zarr "github.com/rkm/zarr-go"
	"github.com/rkm/zarr-go/store/fs"
)

func main() {
	ctx := context.Background()

	st, err := fs.New("/data/example.zarr")
	if err != nil { panic(err) }

	a, err := zarr.OpenArray[float32](ctx, st, "")
	if err != nil { panic(err) }

	// Read the top-left 1024×1024 tile of a 2-D array.
	sel := zarr.Selection{Ranges: []zarr.DimRange{
		{Start: 0, Stop: 1024, Step: 1},
		{Start: 0, Stop: 1024, Step: 1},
	}}
	out := make([]float32, 1024*1024)
	if err := a.Read(ctx, sel, out); err != nil {
		panic(err)
	}
	fmt.Println("first element:", out[0])
}
```

`zarr.All(shape)` returns a selection covering the whole array.

### Write

```go
package main

import (
	"context"
	"encoding/json"

	zarr "github.com/rkm/zarr-go"
	"github.com/rkm/zarr-go/codec"
	"github.com/rkm/zarr-go/store/fs"
)

func main() {
	ctx := context.Background()
	st, _ := fs.New("/data/out.zarr")

	shape := []int64{4096, 4096}
	chunk := []int64{512, 512}

	a, err := zarr.CreateArray[float32](
		ctx, st, "",
		shape, chunk,
		zarr.WithCodecs(
			codec.Spec{Name: "bytes", Configuration: json.RawMessage(`{"endian":"little"}`)},
			codec.Spec{Name: "zstd",  Configuration: json.RawMessage(`{"level":3}`)},
		),
	)
	if err != nil { panic(err) }

	data := make([]float32, shape[0]*shape[1])
	// ... fill data ...
	if err := a.Write(ctx, zarr.All(shape), data); err != nil {
		panic(err)
	}
}
```

## Stores

| Backend                          | Package                              | Notes                                   |
|----------------------------------|--------------------------------------|-----------------------------------------|
| Local filesystem                 | `store/fs`                           | `fs.WithMmap()` for zero-copy reads     |
| In-memory                        | `store/mem`                          | Tests, transient pipelines              |
| HTTP (read-only)                 | `store/http`                         | Range GETs                              |
| S3                               | `store/s3`                           | aws-sdk-go-v2; If-None-Match for atomic create |
| Google Cloud Storage             | `store/gcs`                          |                                         |
| Azure Blob                       | `store/azblob`                       |                                         |
| LRU cache decorator              | `store/cache`                        | Bytes + entry caps, singleflight        |

Stores all satisfy `store.Store`; writable backends additionally satisfy
`store.WritableStore`. A backend that supports atomic creates implements
`SetIfNotExists`; everything else falls back to `Set`.

```go
import (
    "github.com/rkm/zarr-go/store/cache"
    "github.com/rkm/zarr-go/store/s3"
)

base, _ := s3.New(ctx, s3.Config{Bucket: "my-bucket", Prefix: "datasets/"})
st := cache.New(cache.Config{
    Underlying: base,
    MaxBytes:   512 << 20, // 512 MiB
    MaxEntries: 4096,
})
```

## Codecs

All v3 core codecs plus the optional `blosc` codec (CGO, opt-in) ship in
sub-packages of `codec/`. Register a custom codec with
`codec.Register(name, factory)` — see `codec_register.go` for the defaults.

The pipeline is built from a metadata `codecs` array; partial decode is
automatic when the array-bytes codec implements `codec.PartialDecoder`
(currently `sharding_indexed`).

### Decompression cap

`internal/declimit` enforces a global cap on decompressed output for the
`gzip`, `zlib`, and `zstd` codecs. Default 1 GiB; tune with:

```go
import "github.com/rkm/zarr-go/internal/declimit"

declimit.SetMax(8 << 30) // 8 GiB
declimit.SetMax(0)        // disable
```

## Concurrency

`Array[T].Read` is internally concurrent and safe to call from multiple
goroutines. `Array[T].Write` does not lock at the chunk level — concurrent
writers that touch the same chunk can lose updates, and concurrent
read+write may observe torn state. Serialize at the application layer if
needed (one writer per array, or a per-chunk mutex).

Tunables:

```go
zarr.OpenArray[float32](ctx, st, "",
    zarr.WithConcurrency(8),    // max in-flight decode workers
    zarr.WithIOConcurrency(16), // max in-flight store fetches
    zarr.WithRangeGap(4 << 20), // sharded range-coalescing gap
)
```

## Error sentinels

| Sentinel                          | Meaning                                            |
|-----------------------------------|----------------------------------------------------|
| `store.ErrKeyNotFound`            | Key absent (wraps `os.ErrNotExist`).               |
| `store.ErrAlreadyExists`          | `CreateArray` / `CreateGroup` lost a race (wraps `os.ErrExist`). Pass `WithOverwrite()` to clobber. |
| `crc32c.ErrChecksumMismatch`      | Stored CRC32C trailer disagrees with body.         |

All work with `errors.Is`.

## Versions

- **v3 read/write.** Full pipeline, sharding (both `index_location=end` and
  `index_location=start`), CRC32C, `dimension_names`, attribute round-trip,
  unknown-field preservation in `ArrayMeta.Extra` / `GroupMeta.Extra`.
- **v2 read.** `.zarray` / `.zgroup` / `.zattrs` are honored when no
  `zarr.json` is present at the path.

## Testing

```bash
go build ./...
go test ./...
go test -race ./...
```

Cloud-backend integration tests are skipped unless their respective
credential env vars are set; see each `store/<backend>/*_test.go`.

The Python conformance fixtures under `testdata/gen/` are generated with
`zarr-python` (`testdata/gen/generate.py` for v3, `generate_v2.py` for v2).

## Contributing

PRs welcome. Please run `go test ./...` and `go vet ./...` before submitting,
and prefer adding a test that demonstrates the bug or the new behavior.

## License

[MIT](LICENSE).
