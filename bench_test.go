package zarr_test

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"

	"github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/codec"
	gzipcodec "github.com/exergy-dev/zarr-go/codec/gzip"
	"github.com/exergy-dev/zarr-go/codec/zstd"
	"github.com/exergy-dev/zarr-go/store/mem"
)

// --- Local helpers (copies of test helpers; *_test.go symbols are not exported). ---

func benchElemBytes(dt string) int {
	switch dt {
	case "bool", "int8", "uint8":
		return 1
	case "int16", "uint16", "float16":
		return 2
	case "int32", "uint32", "float32":
		return 4
	case "int64", "uint64", "float64", "complex64":
		return 8
	case "complex128":
		return 16
	}
	return 0
}

func benchChunkCoords(shape, chunks []int64) [][]int64 {
	rank := len(shape)
	dims := make([]int64, rank)
	for i := range shape {
		dims[i] = (shape[i] + chunks[i] - 1) / chunks[i]
	}
	var out [][]int64
	cur := make([]int64, rank)
	for {
		c := make([]int64, rank)
		copy(c, cur)
		out = append(out, c)
		d := rank - 1
		for d >= 0 {
			cur[d]++
			if cur[d] < dims[d] {
				break
			}
			cur[d] = 0
			d--
		}
		if d < 0 {
			return out
		}
	}
}

func benchExtractRowMajor(arrBytes []byte, arrShape []int64, coord, chunkShape, actualChunk []int64, elemSize int) []byte {
	rank := len(arrShape)
	arrStrides := make([]int64, rank)
	arrStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		arrStrides[i] = arrStrides[i+1] * arrShape[i+1]
	}
	chunkStrides := make([]int64, rank)
	chunkStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		chunkStrides[i] = chunkStrides[i+1] * actualChunk[i+1]
	}
	total := int64(1)
	for _, n := range actualChunk {
		total *= n
	}
	out := make([]byte, total*int64(elemSize))
	idx := make([]int64, rank)
	for k := int64(0); k < total; k++ {
		var srcEl, dstEl int64
		for d := 0; d < rank; d++ {
			srcEl += (coord[d]*chunkShape[d] + idx[d]) * arrStrides[d]
			dstEl += idx[d] * chunkStrides[d]
		}
		copy(out[dstEl*int64(elemSize):(dstEl+1)*int64(elemSize)],
			arrBytes[srcEl*int64(elemSize):(srcEl+1)*int64(elemSize)])
		for d := rank - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < actualChunk[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out
}

func benchItoa(n int64) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

func benchGzipBytes(b *testing.B, in []byte) []byte {
	b.Helper()
	var buf bytes.Buffer
	w, err := gzip.NewWriterLevel(&buf, 5)
	if err != nil {
		b.Fatal(err)
	}
	if _, err := w.Write(in); err != nil {
		b.Fatal(err)
	}
	if err := w.Close(); err != nil {
		b.Fatal(err)
	}
	return buf.Bytes()
}

// benchWriteArray builds a fresh mem store backing a v3 array.
// If gzipChunks is true, chunk payloads are gzip-encoded (level 5).
func benchWriteArray(b *testing.B, shape, chunks []int64, dtype string, data []byte, gzipChunks bool) *mem.Store {
	b.Helper()
	codecs := []map[string]any{
		{"name": "bytes", "configuration": map[string]any{"endian": "little"}},
	}
	if gzipChunks {
		codecs = append(codecs, map[string]any{
			"name":          "gzip",
			"configuration": map[string]any{"level": 5},
		})
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
		b.Fatal(err)
	}
	s := mem.New()
	ctx := context.Background()
	if err := s.Set(ctx, "arr/zarr.json", raw); err != nil {
		b.Fatal(err)
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
		buf := benchExtractRowMajor(data, shape, c, chunks, actualChunk, es)
		if gzipChunks {
			buf = benchGzipBytes(b, buf)
		}
		key := "arr/c"
		for _, ci := range c {
			key += "/" + benchItoa(ci)
		}
		if err := s.Set(ctx, key, buf); err != nil {
			b.Fatal(err)
		}
	}
	return s
}

func makeFloat32Bytes(n int) []byte {
	raw := make([]byte, n*4)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(float32(i)*0.5))
	}
	return raw
}

func makeInt16Bytes(n int) []byte {
	raw := make([]byte, n*2)
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint16(raw[i*2:], uint16(i))
	}
	return raw
}

// --- Benchmarks -----------------------------------------------------------

func BenchmarkReadFloat32_Uncompressed_2D(b *testing.B) {
	shape := []int64{256, 256}
	chunks := []int64{64, 64}
	n := int(shape[0] * shape[1])
	raw := makeFloat32Bytes(n)
	s := benchWriteArray(b, shape, chunks, "float32", raw, false)
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

func BenchmarkReadFloat32_Gzip_2D(b *testing.B) {
	shape := []int64{256, 256}
	chunks := []int64{64, 64}
	n := int(shape[0] * shape[1])
	raw := makeFloat32Bytes(n)
	s := benchWriteArray(b, shape, chunks, "float32", raw, true)
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

func BenchmarkReadInt16_Uncompressed_3D(b *testing.B) {
	shape := []int64{32, 32, 32}
	chunks := []int64{16, 16, 16}
	n := int(shape[0] * shape[1] * shape[2])
	raw := makeInt16Bytes(n)
	s := benchWriteArray(b, shape, chunks, "int16", raw, false)
	ctx := context.Background()
	a, err := zarr.OpenArray[int16](ctx, s, "arr")
	if err != nil {
		b.Fatal(err)
	}
	sel := zarr.All(shape)
	dst := make([]int16, n)
	b.SetBytes(int64(n) * 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := a.Read(ctx, sel, dst); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkReadFloat32_Slice_2D(b *testing.B) {
	shape := []int64{256, 256}
	chunks := []int64{64, 64}
	n := int(shape[0] * shape[1])
	raw := makeFloat32Bytes(n)
	s := benchWriteArray(b, shape, chunks, "float32", raw, false)
	ctx := context.Background()
	a, err := zarr.OpenArray[float32](ctx, s, "arr")
	if err != nil {
		b.Fatal(err)
	}
	// Centered 100x100 sub-slice: rows [78,178), cols [78,178).
	sel := zarr.Selection{
		{Start: 78, Stop: 178, Step: 1},
		{Start: 78, Stop: 178, Step: 1},
	}
	sliceN := int(sel.NumElements())
	dst := make([]float32, sliceN)
	b.SetBytes(int64(sliceN) * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := a.Read(ctx, sel, dst); err != nil {
			b.Fatal(err)
		}
	}
}

// benchmarkWrite measures Array.Write of a whole 256x256 float32 array in
// 64x64 chunks (16 chunks per Write) into an in-memory store.
func benchmarkWrite(b *testing.B, specs ...codec.Spec) {
	shape := []int64{256, 256}
	chunks := []int64{64, 64}
	n := int(shape[0] * shape[1])
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i) * 0.5
	}
	ctx := context.Background()
	a, err := zarr.CreateArray[float32](ctx, mem.New(), "arr", shape, chunks, zarr.WithCodecs(specs...))
	if err != nil {
		b.Fatal(err)
	}
	sel := zarr.All(shape)
	b.SetBytes(int64(n) * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := a.Write(ctx, sel, src); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkWriteFloat32_Uncompressed_2D(b *testing.B) { benchmarkWrite(b) }
func BenchmarkWriteFloat32_Gzip_2D(b *testing.B)         { benchmarkWrite(b, gzipcodec.Spec(5)) }
func BenchmarkWriteFloat32_Zstd_2D(b *testing.B)         { benchmarkWrite(b, zstd.Spec(3)) }
