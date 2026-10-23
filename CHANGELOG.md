# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is pre-1.0; the public API may change between minor versions
without notice until v1.0.0.

## [Unreleased]

### Changed

- The `store/gcs` and `store/azblob` modules now require Go 1.26 or newer:
  the patched `golang.org/x/net` releases their SDKs depend on need it. The
  root module and `store/s3` still build with Go 1.25.

## [0.1.0] - 2026-10-23

Initial release.

### Added

- Zarr v2 and v3 array read/write support, including sharded chunks
  (`sharding_indexed`), partial decode of sharded chunks (fetches only the
  inner chunks a read actually intersects, not the whole shard), and
  rank-0 (scalar) arrays.
- Codec pipeline: `bytes`, `gzip`, `zlib`, `zstd` (pure Go), `blosc`
  (cgo-only, via the opt-in `zarrcgo` build tag), `transpose`, `crc32c`,
  `sharding_indexed`.
- Store backends: local filesystem (atomic creates via
  `CreateTemp`+fsync+rename/link, optional mmap reads), in-memory, HTTP
  (read-only), S3, Google Cloud Storage, Azure Blob Storage, and an LRU
  cache decorator that coalesces concurrent misses to a single underlying
  fetch. All backends share one `store.Store` contract and conformance
  suite (`store/storetest`).
- `zarr.SetMaxDecodedBytes` / `MaxDecodedBytes`: a process-wide cap on
  decompressed chunk size, enforced by every compression codec, to bound
  memory use against a decompression-bomb-shaped chunk.
- `zarr.WithIOConcurrency` and `zarr.WithRangeGap` to tune, respectively,
  the max in-flight store requests and the byte-gap under which adjacent
  sharded-chunk ranges are coalesced into one fetch.
- `zarr.WithOverwrite`, `zarr.WithChunkKeySeparator`,
  `zarr.WithV2ChunkKeys` array-creation options.
- `Array.Attrs` and `Array.DimensionNames` expose (copies of) what
  `WithAttributes` / `WithDimensionNames` persist, and a v2 array's
  `.zattrs`. `CreateArray` rejects a dimension-name list whose length
  differs from the array rank.
- `store.ErrInvalidKey`, wrapped by every client-side key rejection: an
  empty key on the S3, GCS and Azure backends (previously an internal,
  unmatchable error), and a `..` segment, reserved staging-file segment or
  non-local path on the filesystem and HTTP stores.
- Fuzz targets for the N-d copy kernel (`internal/ndcopy`), sharded index
  trailer parsing, sharded partial decode, and v2/v3 metadata decoding.
- A `Spec` constructor in every codec package (`gzip.Spec(level)`,
  `zstd.Spec(level)`, `zlib.Spec(level)`, `crc32c.Spec()`,
  `transpose.Spec(order...)`, `bytes.Spec(endian)` and
  `sharding.Spec(chunkShape, codecs...)`) returning the `codec.Spec` for
  `zarr.WithCodecs`, in place of hand-written configuration JSON.
- `codec.EnsureArrayToBytes` and `codec.DefaultArrayToBytes`: the rule
  `WithCodecs` and `sharding.Spec` use to complete a codec list that names
  no array→bytes codec.
- `Array.Path` and `Array.FillValue`.
- Runnable examples (`example_test.go`) and `Write` benchmarks.

### Changed

- `zarr.Selection` is a slice of `DimRange` (`zarr.Selection{{Start: 0,
  Stop: 1024}, {Start: 0, Stop: 1024}}`) instead of a struct with a
  `Ranges` field, and a `DimRange` with `Step` 0 selects every element,
  like `Step` 1. `Validate` rejects a negative step.
- `WithCodecs` inserts a little-endian `bytes` codec (in front of the first
  bytes→bytes codec) when the list has no array→bytes codec, so
  `WithCodecs(zstd.Spec(3))` is a complete pipeline.
- `WithFillValue` takes `any`: a Go value of the array's element kind
  (bool, integer, float including NaN and ±Inf, `Float16`, complex), or a
  `json.RawMessage` in the v3 form as before. Encoding errors are reported
  by `CreateArray`.
- `Array.Write` encodes and stores the chunks a call touches concurrently,
  bounded by `WithConcurrency` and `WithIOConcurrency` like `Read`
  (previously one chunk at a time). Its documented caveat is unchanged:
  concurrent `Write` calls that touch the same chunk are not coordinated.
- The zstd codec's encoder is shared process-wide per (level, checksum)
  configuration, built on first use and sized for GOMAXPROCS concurrent
  encodes, instead of one four-slot encoder per open array (each slot
  preallocates match tables, so the per-array encoders scaled memory with
  the number of arrays open).
- `fs.New` creates the root directory (and missing parents) instead of
  requiring it to exist.
- The release procedure moved from this file to `RELEASING.md`.
- Codecs are constructed in one step: `codec.Factory` is
  `func(config json.RawMessage) (Codec, error)` and `Codec.Configure` is
  gone. Every codec package has a `New(config)` constructor, and codecs
  must be safe for concurrent use (a pipeline is shared by all of an
  array's reader goroutines). Only relevant to custom codec
  implementations. The cgo zstd variant is removed.
- The zstd codec decodes with up to GOMAXPROCS concurrent decoders
  (previously one per array, which serialized every read) and encodes with
  up to min(4, GOMAXPROCS).
- The S3, GCS and Azure Blob stores are separate modules:
  `github.com/exergy-dev/zarr-go/store/s3`, `.../store/gcs` and
  `.../store/azblob`, each with its own `go.mod` (import paths are
  unchanged). The root module does not require the AWS, Google Cloud or
  Azure SDKs, nor what they pull in (gRPC, OpenTelemetry, genproto,
  `golang.org/x/crypto`, ...), so a program using only the local,
  in-memory, HTTP or cache stores carries none of them in its module graph
  or its `govulncheck` report. Add a cloud store with `go get` on its
  module path. The `fs` store's mmap path uses the standard library's
  `syscall.Mmap` rather than `golang.org/x/sys/unix`.
- `CreateGroup` / `Group.CreateGroup` no longer take a positional `attrs`
  map: pass `WithAttributes` instead. An array-only option (`WithCodecs`,
  `WithFillValue`, `WithDimensionNames`, `WithChunkKeySeparator`,
  `WithV2ChunkKeys`) passed to `CreateGroup` is now an error instead of
  being silently ignored.
- `CreateOption` is now an interface, and every `OpenOption`
  (`WithConcurrency`, `WithIOConcurrency`, `WithRangeGap`) is also a
  `CreateOption`, so `CreateArray` can tune the array it returns.
- `DecodeError` no longer has a `Codec` field: it always held the
  array->bytes codec's name, even when a bytes->bytes codec such as gzip
  failed. The wrapped `Err` now names the codec that actually failed.
- `codec.PartialDecoder.DecodePartial` takes a `*codec.PartialReadRequest`
  instead of seven positional parameters, and `codec.RangeFetcher` now
  includes `GetSuffix` (`codec.SuffixRangeFetcher` is gone), so every
  `store.Store` is a `RangeFetcher`. Only relevant to custom
  `codec.PartialDecoder` implementations, not typical `Array[T]` callers.
- `GetSuffix` is part of `store.Store` (`store.SuffixReader` is gone), and
  the Azure Blob backend now implements it (size from GetProperties, then
  a ranged download). The sharding codec always reads a trailing index
  with one `GetSuffix`; its fetch-the-whole-shard fallback for stores
  without one is gone.
- `store.Store` is now read-only (`Get`, `GetRange`, `GetSuffix`,
  `Exists`, `List`);
  the new `store.Writer` adds `Set`, `Delete` and `SetIfNotExists` and
  replaces `store.WritableStore`. The HTTP store no longer has stub
  mutators, and `http.ErrReadOnly` is gone: the single sentinel is
  `store.ErrReadOnly`, which `zarr.ErrReadOnly` now aliases, so
  `CreateArray`, `CreateGroup` and `Array.Write` on a read-only store
  (previously failing with `http.ErrReadOnly` or a wrapped write error)
  return an error matching `zarr.ErrReadOnly`.
- `cache.New(underlying, cfg)` returns a read-only `*cache.Store`;
  `cache.NewWriter(underlying, cfg)` wraps a `store.Writer` and returns a
  `*cache.Writer`. `cache.Config` no longer has an `Underlying` field.
- `zarr.Numeric` no longer permits defined/derived types (dropped `~` from
  the constraint): `DtypeOf` and the byte-level codecs switch on a type
  parameter's concrete type, so a derived type (e.g. `type MyFloat32
  float32`) previously satisfied the constraint but silently resolved to
  `DtypeInvalid` instead of being a compile error.

### Fixed

- A shard index whose (offset, length) entries overflow no longer panics
  the partial-decode path (which runs in a `Read` goroutine and so took
  down the process); such entries, and inner chunks that decode to the
  wrong size, are rejected with an error.
- `fs.Store.GetRange` no longer panics when `off+n` overflows, and the S3,
  GCS and Azure stores clamp such ranges instead of sending a malformed
  Range header.
- Node paths are normalized (`"a/"`, `"/a"` and `"a"` are the same node):
  previously a leading or trailing slash made every chunk key miss, so
  reads silently returned fill values. Root-array chunk keys no longer
  start with `/`.
- The cache no longer serves or stores a value that predates a completed
  `Set`/`Delete`, and a value larger than `MaxBytes` no longer evicts the
  whole cache.
- Cloud `List` with a store prefix no longer yields objects outside it
  (e.g. `data2/...` for prefix `data`); an empty key is rejected instead
  of addressing the object named like the prefix.
- Integer fill values are range-checked against the dtype; v2 integer fill
  values keep full 64-bit precision; float16 fill values round to
  nearest-even; fill bytes and the shard index table use native byte
  order.
- The filesystem store publishes files with mode 0644 (was 0600), rejects
  keys that escape its root on Windows, and `Exists` is false for
  directories.
- The HTTP store bounds response bodies to the requested range, redacts
  credentials in error messages, and drains bodies so connections are
  reused.

### Notes

- Breaking changes are possible between 0.x releases as the API is refined
  toward v1.0.0.
