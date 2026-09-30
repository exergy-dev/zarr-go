package zarr_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/store/fs"
)

// writeArrayToFS materializes the same shape/chunks/data layout used by
// benchWriteArray, but onto a t.TempDir() via the fs backend.
func writeArrayToFS(tb testing.TB, root string, shape, chunks []int64, dtype string, data []byte) {
	tb.Helper()
	codecs := []map[string]any{
		{"name": "bytes", "configuration": map[string]any{"endian": "little"}},
	}
	meta := map[string]any{
		"zarr_format": 3,
		"node_type":   "array",
		"shape":       shape,
		"data_type":   dtype,
		"chunk_grid": map[string]any{
			"name":          "regular",
			"configuration": map[string]any{"chunk_shape": chunks},
		},
		"chunk_key_encoding": map[string]any{
			"name":          "default",
			"configuration": map[string]any{"separator": "/"},
		},
		"fill_value": 0,
		"codecs":     codecs,
	}
	raw, err := json.Marshal(meta)
	if err != nil {
		tb.Fatal(err)
	}
	s, err := fs.New(root)
	if err != nil {
		tb.Fatal(err)
	}
	ctx := context.Background()
	if err := s.Set(ctx, "arr/zarr.json", raw); err != nil {
		tb.Fatal(err)
	}
	rank := len(shape)
	es := benchElemBytes(dtype)
	for _, c := range benchChunkCoords(shape, chunks) {
		actualChunk := make([]int64, rank)
		for i, cs := range chunks {
			origin := c[i] * cs
			end := origin + cs
			if end > shape[i] {
				end = shape[i]
			}
			actualChunk[i] = end - origin
		}
		// On-disk chunks are always the full chunk_shape (v3 spec); edge
		// chunks are zero-padded.
		buf := make([]byte, chunks[0]*chunks[1]*int64(es))
		full := benchExtractRowMajor(data, shape, c, chunks, actualChunk, es)
		if int64(len(full)) == int64(len(buf)) {
			copy(buf, full)
		} else {
			// rank>2 or partial — copy via the full extractor at chunk shape.
			fullChunk := benchExtractRowMajor(data, shape, c, chunks, chunks, es)
			copy(buf, fullChunk)
		}
		key := "arr/c"
		for _, ci := range c {
			key += "/" + benchItoa(ci)
		}
		if err := s.Set(ctx, key, buf); err != nil {
			tb.Fatal(err)
		}
	}
}

func TestFSReadWithMmapMatchesBaseline(t *testing.T) {
	shape := []int64{128, 128}
	chunks := []int64{32, 32}
	n := int(shape[0] * shape[1])
	raw := makeFloat32Bytes(n)
	dir := t.TempDir()
	writeArrayToFS(t, dir, shape, chunks, "float32", raw)

	ctx := context.Background()

	// Baseline: no-mmap fs store.
	sNoMmap, err := fs.New(dir)
	if err != nil {
		t.Fatal(err)
	}
	a1, err := zarr.OpenArray[float32](ctx, sNoMmap, "arr")
	if err != nil {
		t.Fatal(err)
	}
	dst1 := make([]float32, n)
	if err := a1.Read(ctx, zarr.All(shape), dst1); err != nil {
		t.Fatal(err)
	}

	// Mmap-enabled store: must produce identical results.
	sMmap, err := fs.New(dir, fs.WithMmap())
	if err != nil {
		t.Fatal(err)
	}
	a2, err := zarr.OpenArray[float32](ctx, sMmap, "arr")
	if err != nil {
		t.Fatal(err)
	}
	dst2 := make([]float32, n)
	if err := a2.Read(ctx, zarr.All(shape), dst2); err != nil {
		t.Fatal(err)
	}

	for i := range dst1 {
		if dst1[i] != dst2[i] {
			t.Fatalf("mmap divergence at %d: no-mmap=%v mmap=%v", i, dst1[i], dst2[i])
		}
	}
}

// benchmarkFSRead runs Read(All) over a 256x256 float32 array stored on disk.
// useMmap toggles the WithMmap() option on the fs store.
func benchmarkFSRead(b *testing.B, useMmap bool) {
	b.Helper()
	shape := []int64{256, 256}
	chunks := []int64{64, 64}
	n := int(shape[0] * shape[1])
	raw := makeFloat32Bytes(n)
	dir := b.TempDir()
	writeArrayToFS(b, dir, shape, chunks, "float32", raw)

	var (
		s   *fs.Store
		err error
	)
	if useMmap {
		s, err = fs.New(dir, fs.WithMmap())
	} else {
		s, err = fs.New(dir)
	}
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	a, err := zarr.OpenArray[float32](ctx, s, "arr")
	if err != nil {
		b.Fatal(err)
	}
	sel := zarr.All(shape)
	dst := make([]float32, n)
	b.SetBytes(int64(n) * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := a.Read(ctx, sel, dst); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFSReadFloat32_NoMmap(b *testing.B) { benchmarkFSRead(b, false) }
func BenchmarkFSReadFloat32_Mmap(b *testing.B)   { benchmarkFSRead(b, true) }

// benchmarkFSReadLarge exercises the case mmap was designed to win: large
// chunks (1 MiB+) where mmap saves the read-into-heap copy. Real NVMe-backed
// scientific datasets routinely use chunks in this range.
func benchmarkFSReadLarge(b *testing.B, useMmap bool) {
	b.Helper()
	shape := []int64{2048, 2048}
	chunks := []int64{512, 512}
	n := int(shape[0] * shape[1])
	raw := makeFloat32Bytes(n)
	dir := b.TempDir()
	writeArrayToFS(b, dir, shape, chunks, "float32", raw)

	var (
		s   *fs.Store
		err error
	)
	if useMmap {
		s, err = fs.New(dir, fs.WithMmap())
	} else {
		s, err = fs.New(dir)
	}
	if err != nil {
		b.Fatal(err)
	}
	ctx := context.Background()
	a, err := zarr.OpenArray[float32](ctx, s, "arr")
	if err != nil {
		b.Fatal(err)
	}
	sel := zarr.All(shape)
	dst := make([]float32, n)
	b.SetBytes(int64(n) * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := a.Read(ctx, sel, dst); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFSReadFloat32_Large_NoMmap(b *testing.B) { benchmarkFSReadLarge(b, false) }
func BenchmarkFSReadFloat32_Large_Mmap(b *testing.B)   { benchmarkFSReadLarge(b, true) }
