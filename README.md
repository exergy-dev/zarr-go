# zarr-go

[![CI](https://github.com/exergy-dev/zarr-go/actions/workflows/ci.yml/badge.svg)](https://github.com/exergy-dev/zarr-go/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/exergy-dev/zarr-go.svg)](https://pkg.go.dev/github.com/exergy-dev/zarr-go)
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
go get github.com/exergy-dev/zarr-go
```

Requires Go 1.25 or newer.

The S3, GCS and Azure Blob stores are separate modules, so their SDKs only
enter your build if you use them. Add the ones you need:

```bash
go get github.com/exergy-dev/zarr-go/store/s3
go get github.com/exergy-dev/zarr-go/store/gcs
go get github.com/exergy-dev/zarr-go/store/azblob
```

The import paths are the same as the module paths. Every other store
(`fs`, `mem`, `http`, `cache`) ships in the root module.

## Quick start

### Read

```go
package main

import (
	"context"
	"fmt"

	zarr "github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/store/fs"
)

func main() {
	ctx := context.Background()

	st, err := fs.New("/data/example.zarr")
	if err != nil { panic(err) }

	a, err := zarr.OpenArray[float32](ctx, st, "")
	if err != nil { panic(err) }

	// Read the top-left 1024×1024 tile of a 2-D array.
	sel := zarr.Selection{{Start: 0, Stop: 1024}, {Start: 0, Stop: 1024}}
	out := make([]float32, sel.NumElements())
	if err := a.Read(ctx, sel, out); err != nil {
		panic(err)
	}
	fmt.Println("first element:", out[0])
}
```

A `Selection` has one `[Start, Stop)` range per dimension; `Step` strides
(0 and 1 both mean every element). `zarr.All(shape)` covers the whole
array. `Open` (instead of `OpenArray[T]`) returns a `Node` to type-switch
on when the dtype is only known at run time, and `Array.Stream` yields
chunks one at a time when the selection would not fit in memory.

### Write

```go
package main

import (
	"context"
	"math"

	zarr "github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/codec/zstd"
	"github.com/exergy-dev/zarr-go/store/fs"
)

func main() {
	ctx := context.Background()
	st, err := fs.New("/data/out.zarr") // created if missing
	if err != nil { panic(err) }

	shape := []int64{4096, 4096}
	chunk := []int64{512, 512}

	a, err := zarr.CreateArray[float32](ctx, st, "", shape, chunk,
		zarr.WithCodecs(zstd.Spec(3)),
		zarr.WithFillValue(float32(math.NaN())),
	)
	if err != nil { panic(err) }

	data := make([]float32, shape[0]*shape[1])
	// ... fill data ...
	if err := a.Write(ctx, zarr.All(shape), data); err != nil {
		panic(err)
	}
}
```

`WithCodecs` takes the pipeline in metadata order; when it has no
array→bytes codec, a little-endian `bytes` codec is inserted, so a bare
compressor is enough. `WithFillValue` takes a Go value of the element kind
(NaN and ±Inf included) or a `json.RawMessage` in the v3 form. More
runnable examples are in `example_test.go`.

## Stores

| Backend                          | Package                              | Notes                                   |
|----------------------------------|--------------------------------------|-----------------------------------------|
| Local filesystem                 | `store/fs`                           | `fs.WithMmap()` for zero-copy reads     |
| In-memory                        | `store/mem`                          | Tests, transient pipelines              |
| HTTP (read-only)                 | `store/http`                         | Range GETs                              |
| S3                               | `store/s3` (own module)              | aws-sdk-go-v2; If-None-Match for atomic create |
| Google Cloud Storage             | `store/gcs` (own module)             |                                         |
| Azure Blob                       | `store/azblob` (own module)          |                                         |
| LRU cache decorator              | `store/cache`                        | Bytes + entry caps, coalesces concurrent misses |

Every store satisfies `store.Store`, the read side: `Get`, `GetRange`,
`GetSuffix` (the last n bytes, which is how a sharded read fetches a
shard's index), `Exists` and `List`. Writable backends also satisfy
`store.Writer` (`Set`, `Delete`, `SetIfNotExists`); `CreateArray`,
`CreateGroup` and `Array.Write` return `zarr.ErrReadOnly` (the same value
as `store.ErrReadOnly`) for a store that isn't one, such as the HTTP
store. Keys a backend refuses to address (empty, or escaping its root)
fail with `store.ErrInvalidKey`.

`cache.New` returns a read-only cache; `cache.NewWriter` wraps a
`store.Writer`, writes through, and invalidates the written keys.

```go
import (
    "github.com/exergy-dev/zarr-go/store/cache"
    "github.com/exergy-dev/zarr-go/store/s3"
)

base, _ := s3.New(ctx, s3.Config{Bucket: "my-bucket", Prefix: "datasets/"})
st := cache.NewWriter(base, cache.Config{
    MaxBytes:   512 << 20, // 512 MiB
    MaxEntries: 4096,
})
```

## Codecs

All v3 core codecs plus the optional `blosc` codec (CGO, opt-in) ship in
sub-packages of `codec/`, each with a `Spec` constructor for `WithCodecs`:

| Codec              | Spec                                        | Kind          |
|--------------------|---------------------------------------------|---------------|
| `transpose`        | `transpose.Spec(order ...int)`              | array→array   |
| `bytes`            | `bytes.Spec(bytes.Big)` (little is implied) | array→bytes   |
| `sharding_indexed` | `sharding.Spec(innerChunks, codecs...)`     | array→bytes   |
| `gzip`             | `gzip.Spec(level)`                          | bytes→bytes   |
| `zlib`             | `zlib.Spec(level)`                          | bytes→bytes   |
| `zstd`             | `zstd.Spec(level)`                          | bytes→bytes   |
| `crc32c`           | `crc32c.Spec()`                             | bytes→bytes   |
| `blosc`            | `codec.Spec{Name: "blosc", Configuration: ...}` | bytes→bytes |

```go
// 1024×1024 shards of 64×64 zstd-compressed inner chunks, each checksummed.
zarr.CreateArray[float32](ctx, st, "", shape, []int64{1024, 1024},
    zarr.WithCodecs(sharding.Spec([]int64{64, 64}, zstd.Spec(3), crc32c.Spec())))
```

Any `codec.Spec` works, so a configuration a constructor doesn't expose
(zstd checksums, a shard index at the start) is one literal away. Register
a custom codec with `codec.Register(name, factory)`; see `codec_register.go`
for the defaults.

Reading a sub-region of a sharded array fetches only the inner chunks it
intersects (`codec.PartialDecoder`), coalescing nearby byte ranges into one
request.

### Decompression cap

A global cap on decompressed output guards `gzip`, `zlib`, `zstd`, and
`blosc` against decompression bombs — a small encoded chunk that declares
(or expands to) an enormous decoded size. Default 1 GiB; tune with:

```go
zarr.SetMaxDecodedBytes(8 << 30) // 8 GiB
zarr.SetMaxDecodedBytes(0)       // disable
```

## Concurrency

`Read` and `Write` each process the chunks they touch concurrently, and
`Read` is safe to call from multiple goroutines. `Write` does not lock at
the chunk level — two concurrent `Write` calls that touch the same chunk
can lose updates, and a concurrent read may observe a half-written chunk.
Serialize at the application layer if needed (one writer per array, or
disjoint chunk ranges per writer).

Tunables:

```go
zarr.OpenArray[float32](ctx, st, "",
    zarr.WithConcurrency(8),    // max concurrent chunk decodes / encodes
    zarr.WithIOConcurrency(16), // max in-flight store requests
    zarr.WithRangeGap(4 << 20), // sharded range-coalescing gap
)
```

These are `OpenOption`s, and every `OpenOption` is also a `CreateOption`,
so the same calls can be passed to `CreateArray` to tune the array it
returns.

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

`./...` stops at module boundaries, so the commands above skip the nested
`store/s3`, `store/gcs` and `store/azblob` modules. `make test`, `make race`,
`make vet` and `make lint` run over all four modules.

Each cloud backend's offline tests run against an in-process fake of its
object API. Its integration tests are skipped unless the backend's
credential env vars are set; see each `store/<backend>/*_test.go`.

The Python conformance fixtures under `testdata/gen/` are generated with
`zarr-python` (`testdata/gen/generate.py` for v3, `generate_v2.py` for v2).

## Contributing

PRs welcome. Please run `go test ./...` and `go vet ./...` before submitting,
and prefer adding a test that demonstrates the bug or the new behavior.
Changes are recorded in [CHANGELOG.md](CHANGELOG.md); the tagging procedure
for the four modules is in [RELEASING.md](RELEASING.md).

## License

[MIT](LICENSE).
