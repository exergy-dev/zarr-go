# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project is pre-1.0; the public API may change between minor versions
without notice until v1.0.0.

## [0.1.0] - Unreleased

Initial release.

### Added

- Zarr v2 and v3 array read/write support, including sharded chunks
  (`sharding_indexed`), partial decode of sharded chunks (fetches only the
  inner chunks a read actually intersects, not the whole shard), and
  rank-0 (scalar) arrays.
- Codec pipeline: `bytes`, `gzip`, `zlib`, `zstd` (pure-Go by default, cgo
  via the opt-in `zarrcgo` build tag), `blosc` (cgo-only), `transpose`,
  `crc32c`, `sharding_indexed`.
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
  trailer parsing, and v2/v3 metadata decoding.

### Changed

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

### Notes

- Breaking changes are possible between 0.x releases as the API is refined
  toward v1.0.0.
