// Package zarr reads and writes Zarr arrays: chunked, compressed,
// N-dimensional arrays stored as key/value objects. It implements Zarr v3
// (read and write) and reads Zarr v2.
//
// An array lives in a store.Store: a directory (store/fs), memory
// (store/mem), an HTTP server (store/http), or an object store (store/s3,
// store/gcs, store/azblob, each its own module). [OpenArray]
// opens an existing array as an [Array] whose element type T matches its
// dtype; [CreateArray] makes a new one:
//
//	st, err := fs.New("/data/out.zarr")
//	a, err := zarr.CreateArray[float32](ctx, st, "", []int64{4096, 4096}, []int64{512, 512},
//		zarr.WithCodecs(zstd.Spec(3)))
//	err = a.Write(ctx, zarr.All(a.Shape()), data)
//
// [Array.Read] and [Array.Write] move a [Selection], a strided
// hyperrectangle, between the array and a flat row-major slice, fetching
// or storing the chunks it touches concurrently. [Array.Stream] yields
// chunks one at a time for data that does not fit in memory.
//
// Compression and other codecs are configured by [WithCodecs] from the
// Spec constructor in each codec package (gzip.Spec, zstd.Spec,
// sharding.Spec, ...). Every codec in the v3 specification ships in a
// subpackage of codec and is registered by importing this package.
package zarr
