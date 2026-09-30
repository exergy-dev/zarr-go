package zarr_test

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"

	"github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/store/mem"
)

// writeArray writes a synthetic v3 array into a fresh mem store and returns it.
// Codecs default to a single bytes-only pipeline.
func writeArray(t *testing.T, shape, chunks []int64, dtype string, data []byte, codecs []map[string]any) *mem.Store {
	t.Helper()
	if codecs == nil {
		codecs = []map[string]any{{"name": "bytes", "configuration": map[string]any{"endian": "little"}}}
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
	raw, _ := json.Marshal(meta)
	s := mem.New()
	ctx := context.Background()
	if err := s.Set(ctx, "arr/zarr.json", raw); err != nil {
		t.Fatal(err)
	}
	// Write chunks at full chunk_shape (v3 spec); edge chunks are padded
	// with zeros (the fill value).
	chunkBytes := elemBytes(dtype)
	for _, c := range chunkCoords(shape, chunks) {
		buf := extractFullChunk(data, shape, c, chunks, chunkBytes)
		key := "arr/c"
		for _, ci := range c {
			key += "/" + itoa(ci)
		}
		if err := s.Set(ctx, key, buf); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

// extractFullChunk returns one full-size chunk (product(chunkShape) * elemSize
// bytes) copied from arrBytes; positions outside the array bounds are zero.
func extractFullChunk(arrBytes []byte, arrShape []int64, coord, chunkShape []int64, elemSize int) []byte {
	rank := len(arrShape)
	arrStrides := make([]int64, rank)
	arrStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		arrStrides[i] = arrStrides[i+1] * arrShape[i+1]
	}
	chunkStrides := make([]int64, rank)
	chunkStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		chunkStrides[i] = chunkStrides[i+1] * chunkShape[i+1]
	}
	full := int64(1)
	for _, n := range chunkShape {
		full *= n
	}
	out := make([]byte, full*int64(elemSize))
	idx := make([]int64, rank)
	for k := int64(0); k < full; k++ {
		// Skip positions outside the array.
		inBounds := true
		var srcEl, dstEl int64
		for d := 0; d < rank; d++ {
			ap := coord[d]*chunkShape[d] + idx[d]
			if ap >= arrShape[d] {
				inBounds = false
			}
			srcEl += ap * arrStrides[d]
			dstEl += idx[d] * chunkStrides[d]
		}
		if inBounds {
			copy(out[dstEl*int64(elemSize):(dstEl+1)*int64(elemSize)],
				arrBytes[srcEl*int64(elemSize):(srcEl+1)*int64(elemSize)])
		}
		for d := rank - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < chunkShape[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out
}

func elemBytes(dt string) int {
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

func chunkCoords(shape, chunks []int64) [][]int64 {
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

func itoa(n int64) string {
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

// --- Tests -----------------------------------------------------------------

func TestReadFullArray1D_Float32(t *testing.T) {
	shape := []int64{16}
	chunks := []int64{4}
	expect := make([]float32, shape[0])
	for i := range expect {
		expect[i] = float32(i) * 0.5
	}
	raw := make([]byte, len(expect)*4)
	for i, v := range expect {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	s := writeArray(t, shape, chunks, "float32", raw, nil)
	a, err := zarr.OpenArray[float32](context.Background(), s, "arr")
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, shape[0])
	if err := a.Read(context.Background(), zarr.All(shape), got); err != nil {
		t.Fatal(err)
	}
	for i, v := range expect {
		if got[i] != v {
			t.Fatalf("[%d] want %v got %v", i, v, got[i])
		}
	}
}

func TestReadSlice2D_Int32(t *testing.T) {
	shape := []int64{8, 8}
	chunks := []int64{4, 4}
	expect := make([]int32, shape[0]*shape[1])
	for i := range expect {
		expect[i] = int32(i)
	}
	raw := make([]byte, len(expect)*4)
	for i, v := range expect {
		binary.LittleEndian.PutUint32(raw[i*4:], uint32(v))
	}
	s := writeArray(t, shape, chunks, "int32", raw, nil)
	a, err := zarr.OpenArray[int32](context.Background(), s, "arr")
	if err != nil {
		t.Fatal(err)
	}
	// Slice [2:6, 1:7]
	sel := zarr.Selection{Ranges: []zarr.DimRange{{Start: 2, Stop: 6, Step: 1}, {Start: 1, Stop: 7, Step: 1}}}
	got := make([]int32, sel.NumElements())
	if err := a.Read(context.Background(), sel, got); err != nil {
		t.Fatal(err)
	}
	k := 0
	for r := int64(2); r < 6; r++ {
		for c := int64(1); c < 7; c++ {
			want := expect[r*shape[1]+c]
			if got[k] != want {
				t.Fatalf("[%d,%d] want %v got %v", r, c, want, got[k])
			}
			k++
		}
	}
}

func TestReadEdgeChunks(t *testing.T) {
	// shape 10, chunks 4 → chunks of [4,4,2]
	shape := []int64{10}
	chunks := []int64{4}
	expect := make([]int16, shape[0])
	for i := range expect {
		expect[i] = int16(100 + i)
	}
	raw := make([]byte, len(expect)*2)
	for i, v := range expect {
		binary.LittleEndian.PutUint16(raw[i*2:], uint16(v))
	}
	s := writeArray(t, shape, chunks, "int16", raw, nil)
	a, err := zarr.OpenArray[int16](context.Background(), s, "arr")
	if err != nil {
		t.Fatal(err)
	}
	got := make([]int16, shape[0])
	if err := a.Read(context.Background(), zarr.All(shape), got); err != nil {
		t.Fatal(err)
	}
	for i := range expect {
		if got[i] != expect[i] {
			t.Fatalf("[%d] want %v got %v", i, expect[i], got[i])
		}
	}
}

func TestReadMissingChunkUsesFillValue(t *testing.T) {
	shape := []int64{8}
	chunks := []int64{4}
	// Write meta only; no chunk keys → all fill (zero).
	meta := map[string]any{
		"zarr_format": 3,
		"node_type":   "array",
		"shape":       shape,
		"data_type":   "float32",
		"chunk_grid": map[string]any{
			"name":          "regular",
			"configuration": map[string]any{"chunk_shape": chunks},
		},
		"chunk_key_encoding": map[string]any{
			"name":          "default",
			"configuration": map[string]any{"separator": "/"},
		},
		"fill_value": 7.5,
		"codecs": []any{map[string]any{
			"name":          "bytes",
			"configuration": map[string]any{"endian": "little"},
		}},
	}
	raw, _ := json.Marshal(meta)
	s := mem.New()
	if err := s.Set(context.Background(), "arr/zarr.json", raw); err != nil {
		t.Fatal(err)
	}
	a, err := zarr.OpenArray[float32](context.Background(), s, "arr")
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, shape[0])
	if err := a.Read(context.Background(), zarr.All(shape), got); err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != 7.5 {
			t.Fatalf("[%d] want 7.5 got %v", i, v)
		}
	}
}

func TestOpenArrayRejectsDtypeMismatch(t *testing.T) {
	shape := []int64{4}
	chunks := []int64{4}
	raw := make([]byte, 16)
	s := writeArray(t, shape, chunks, "float32", raw, nil)
	if _, err := zarr.OpenArray[float64](context.Background(), s, "arr"); err == nil {
		t.Fatal("expected dtype mismatch error")
	}
}

func TestReadGzipCompressed(t *testing.T) {
	shape := []int64{8, 8}
	chunks := []int64{4, 4}
	expect := make([]float32, shape[0]*shape[1])
	for i := range expect {
		expect[i] = float32(i) * 0.25
	}
	raw := make([]byte, len(expect)*4)
	for i, v := range expect {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	codecs := []map[string]any{
		{"name": "bytes", "configuration": map[string]any{"endian": "little"}},
		{"name": "gzip", "configuration": map[string]any{"level": 5}},
	}
	s := writeArray(t, shape, chunks, "float32", raw, codecs)
	// writeArray wrote raw chunk bytes; we need to overwrite with gzip-encoded chunks.
	// Rewrite: open and re-encode each chunk through the pipeline.
	rewriteWithCodecs(t, s, shape, chunks, raw, "float32", codecs)
	a, err := zarr.OpenArray[float32](context.Background(), s, "arr")
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(expect))
	if err := a.Read(context.Background(), zarr.All(shape), got); err != nil {
		t.Fatal(err)
	}
	for i := range expect {
		if got[i] != expect[i] {
			t.Fatalf("[%d] want %v got %v", i, expect[i], got[i])
		}
	}
}

// rewriteWithCodecs re-encodes chunk bytes through the array's pipeline so that
// chunk keys hold properly-encoded payloads (writeArray writes raw bytes, which
// only works for bytes-only pipelines).
func rewriteWithCodecs(t *testing.T, s *mem.Store, shape, chunks []int64, raw []byte, dtype string, _ []map[string]any) {
	t.Helper()
	// We open the array via the library to use its pipeline for encoding.
	// EncodeBB on the Pipeline isn't exposed, so we construct chunks and use
	// the same pipeline by going through writeArray (raw) then compressing.
	// Simplest: write each chunk pre-compressed via stdlib gzip.
	// For tests, use a minimal helper.
	for _, c := range chunkCoords(shape, chunks) {
		buf := extractFullChunk(raw, shape, c, chunks, elemBytes(dtype))
		gz := gzipBytes(t, buf)
		key := "arr/c"
		for _, ci := range c {
			key += "/" + itoa(ci)
		}
		if err := s.Set(context.Background(), key, gz); err != nil {
			t.Fatal(err)
		}
	}
}
