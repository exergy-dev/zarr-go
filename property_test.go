package zarr_test

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"testing"

	"pgregory.net/rapid"

	"github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/store/mem"
)

// --- Local helpers (duplicated from array_test.go to keep this file
// self-contained). ----------------------------------------------------------

func propElemBytes(dt string) int {
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

func propChunkCoords(shape, chunks []int64) [][]int64 {
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

// propExtractFullChunk returns one full-size chunk (product(chunkShape)*elemSize
// bytes) with positions outside arrShape zero-padded. This matches the v3 spec
// which writes edge chunks at full size with the fill value in padded slots.
func propExtractFullChunk(arrBytes []byte, arrShape []int64, coord, chunkShape []int64, elemSize int) []byte {
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

func propItoa(n int64) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

// propWriteArray writes a fresh v3 array (bytes-only codec) into a mem store
// and returns it. fillValue is encoded as JSON so use a numeric type.
func propWriteArray(t *rapid.T, shape, chunks []int64, dtype string, data []byte, fillValue any) *mem.Store {
	t.Helper()
	codecs := []map[string]any{{"name": "bytes", "configuration": map[string]any{"endian": "little"}}}
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
		"fill_value": fillValue,
		"codecs":     codecs,
	}
	raw, err := json.Marshal(meta)
	if err != nil {
		t.Fatal(err)
	}
	s := mem.New()
	ctx := context.Background()
	if err := s.Set(ctx, "arr/zarr.json", raw); err != nil {
		t.Fatal(err)
	}
	if data == nil {
		return s
	}
	_ = len(shape)
	chunkBytes := propElemBytes(dtype)
	for _, c := range propChunkCoords(shape, chunks) {
		buf := propExtractFullChunk(data, shape, c, chunks, chunkBytes)
		key := "arr/c"
		for _, ci := range c {
			key += "/" + propItoa(ci)
		}
		if err := s.Set(ctx, key, buf); err != nil {
			t.Fatal(err)
		}
	}
	return s
}

// drawShape generates rank, shape, chunks, and a row-major selection.
type genSpec struct {
	rank   int
	shape  []int64
	chunks []int64
	sel    zarr.Selection
}

func drawSpec(t *rapid.T) genSpec {
	rank := rapid.IntRange(1, 4).Draw(t, "rank")
	shape := make([]int64, rank)
	chunks := make([]int64, rank)
	for i := 0; i < rank; i++ {
		shape[i] = int64(rapid.IntRange(1, 16).Draw(t, fmt.Sprintf("shape[%d]", i)))
		chunks[i] = int64(rapid.IntRange(1, int(shape[i])).Draw(t, fmt.Sprintf("chunks[%d]", i)))
	}
	ranges := make([]zarr.DimRange, rank)
	for i := 0; i < rank; i++ {
		start := int64(rapid.IntRange(0, int(shape[i])-1).Draw(t, fmt.Sprintf("start[%d]", i)))
		stop := int64(rapid.IntRange(int(start)+1, int(shape[i])).Draw(t, fmt.Sprintf("stop[%d]", i)))
		ranges[i] = zarr.DimRange{Start: start, Stop: stop, Step: 1}
	}
	return genSpec{rank: rank, shape: shape, chunks: chunks, sel: zarr.Selection{Ranges: ranges}}
}

// totalElems returns product(shape).
func totalElems(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

// rowMajorIndex returns the row-major flat index for coord in shape.
func rowMajorIndex(coord, shape []int64) int64 {
	var idx int64
	stride := int64(1)
	for d := len(shape) - 1; d >= 0; d-- {
		idx += coord[d] * stride
		stride *= shape[d]
	}
	return idx
}

// iterRowMajor invokes fn for every coordinate in the given shape (row-major).
func iterRowMajor(shape []int64, fn func(coord []int64)) {
	rank := len(shape)
	if rank == 0 {
		return
	}
	total := totalElems(shape)
	idx := make([]int64, rank)
	for k := int64(0); k < total; k++ {
		fn(idx)
		for d := rank - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
		}
	}
}

// --- Per-dtype property runners --------------------------------------------

func runReadEqualsSliceInt32(t *rapid.T) {
	spec := drawSpec(t)
	n := totalElems(spec.shape)
	src := make([]int32, n)
	for i := range src {
		src[i] = rapid.Int32().Draw(t, fmt.Sprintf("src[%d]", i))
	}
	raw := make([]byte, n*4)
	for i, v := range src {
		binary.LittleEndian.PutUint32(raw[i*4:], uint32(v))
	}
	s := propWriteArray(t, spec.shape, spec.chunks, "int32", raw, 0)
	a, err := zarr.OpenArray[int32](context.Background(), s, "arr")
	if err != nil {
		t.Fatal(err)
	}
	got := make([]int32, spec.sel.NumElements())
	if err := a.Read(context.Background(), spec.sel, got); err != nil {
		t.Fatal(err)
	}
	verifySelection(t, spec, got, src)
}

func runReadEqualsSliceInt64(t *rapid.T) {
	spec := drawSpec(t)
	n := totalElems(spec.shape)
	src := make([]int64, n)
	for i := range src {
		src[i] = rapid.Int64().Draw(t, fmt.Sprintf("src[%d]", i))
	}
	raw := make([]byte, n*8)
	for i, v := range src {
		binary.LittleEndian.PutUint64(raw[i*8:], uint64(v))
	}
	s := propWriteArray(t, spec.shape, spec.chunks, "int64", raw, 0)
	a, err := zarr.OpenArray[int64](context.Background(), s, "arr")
	if err != nil {
		t.Fatal(err)
	}
	got := make([]int64, spec.sel.NumElements())
	if err := a.Read(context.Background(), spec.sel, got); err != nil {
		t.Fatal(err)
	}
	verifySelection(t, spec, got, src)
}

func runReadEqualsSliceFloat32(t *rapid.T) {
	spec := drawSpec(t)
	n := totalElems(spec.shape)
	src := make([]float32, n)
	for i := range src {
		// Avoid NaN so equality comparisons are reliable.
		v := rapid.Float32Range(-1e6, 1e6).Draw(t, fmt.Sprintf("src[%d]", i))
		src[i] = v
	}
	raw := make([]byte, n*4)
	for i, v := range src {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	s := propWriteArray(t, spec.shape, spec.chunks, "float32", raw, 0)
	a, err := zarr.OpenArray[float32](context.Background(), s, "arr")
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, spec.sel.NumElements())
	if err := a.Read(context.Background(), spec.sel, got); err != nil {
		t.Fatal(err)
	}
	verifySelection(t, spec, got, src)
}

func runReadEqualsSliceFloat64(t *rapid.T) {
	spec := drawSpec(t)
	n := totalElems(spec.shape)
	src := make([]float64, n)
	for i := range src {
		v := rapid.Float64Range(-1e9, 1e9).Draw(t, fmt.Sprintf("src[%d]", i))
		src[i] = v
	}
	raw := make([]byte, n*8)
	for i, v := range src {
		binary.LittleEndian.PutUint64(raw[i*8:], math.Float64bits(v))
	}
	s := propWriteArray(t, spec.shape, spec.chunks, "float64", raw, 0)
	a, err := zarr.OpenArray[float64](context.Background(), s, "arr")
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float64, spec.sel.NumElements())
	if err := a.Read(context.Background(), spec.sel, got); err != nil {
		t.Fatal(err)
	}
	verifySelection(t, spec, got, src)
}

func runReadEqualsSliceUint8(t *rapid.T) {
	spec := drawSpec(t)
	n := totalElems(spec.shape)
	src := make([]uint8, n)
	for i := range src {
		src[i] = rapid.Uint8().Draw(t, fmt.Sprintf("src[%d]", i))
	}
	raw := make([]byte, n)
	copy(raw, src)
	s := propWriteArray(t, spec.shape, spec.chunks, "uint8", raw, 0)
	a, err := zarr.OpenArray[uint8](context.Background(), s, "arr")
	if err != nil {
		t.Fatal(err)
	}
	got := make([]uint8, spec.sel.NumElements())
	if err := a.Read(context.Background(), spec.sel, got); err != nil {
		t.Fatal(err)
	}
	verifySelection(t, spec, got, src)
}

// verifySelection walks the selection in row-major order, looks up the matching
// element in the source array, and asserts equality with got[k].
func verifySelection[T comparable](t *rapid.T, spec genSpec, got, src []T) {
	t.Helper()
	outShape := make([]int64, spec.rank)
	for i, r := range spec.sel.Ranges {
		outShape[i] = r.Length()
	}
	srcCoord := make([]int64, spec.rank)
	k := 0
	iterRowMajor(outShape, func(coord []int64) {
		for d := 0; d < spec.rank; d++ {
			r := spec.sel.Ranges[d]
			srcCoord[d] = r.Start + coord[d]*r.Step
		}
		srcIdx := rowMajorIndex(srcCoord, spec.shape)
		want := src[srcIdx]
		if got[k] != want {
			t.Fatalf("mismatch at out=%v src=%v: want %v got %v (shape=%v chunks=%v sel=%+v)",
				coord, srcCoord, want, got[k], spec.shape, spec.chunks, spec.sel.Ranges)
		}
		k++
	})
}

// --- Tests -----------------------------------------------------------------

func TestPropertyReadEqualsSlice(t *testing.T) {
	t.Run("int32", func(t *testing.T) {
		rapid.Check(t, runReadEqualsSliceInt32)
	})
	t.Run("int64", func(t *testing.T) {
		rapid.Check(t, runReadEqualsSliceInt64)
	})
	t.Run("float32", func(t *testing.T) {
		rapid.Check(t, runReadEqualsSliceFloat32)
	})
	t.Run("float64", func(t *testing.T) {
		rapid.Check(t, runReadEqualsSliceFloat64)
	})
	t.Run("uint8", func(t *testing.T) {
		rapid.Check(t, runReadEqualsSliceUint8)
	})
}

func TestPropertyMissingChunksUseFillValue(t *testing.T) {
	t.Run("int32", func(t *testing.T) {
		rapid.Check(t, func(t *rapid.T) {
			spec := drawSpec(t)
			fill := rapid.Int32().Draw(t, "fill")
			s := propWriteArray(t, spec.shape, spec.chunks, "int32", nil, fill)
			a, err := zarr.OpenArray[int32](context.Background(), s, "arr")
			if err != nil {
				t.Fatal(err)
			}
			n := totalElems(spec.shape)
			got := make([]int32, n)
			if err := a.Read(context.Background(), zarr.All(spec.shape), got); err != nil {
				t.Fatal(err)
			}
			for i, v := range got {
				if v != fill {
					t.Fatalf("[%d] want fill %d got %d", i, fill, v)
				}
			}
		})
	})
	t.Run("int64", func(t *testing.T) {
		rapid.Check(t, func(t *rapid.T) {
			spec := drawSpec(t)
			fill := rapid.Int64().Draw(t, "fill")
			s := propWriteArray(t, spec.shape, spec.chunks, "int64", nil, fill)
			a, err := zarr.OpenArray[int64](context.Background(), s, "arr")
			if err != nil {
				t.Fatal(err)
			}
			n := totalElems(spec.shape)
			got := make([]int64, n)
			if err := a.Read(context.Background(), zarr.All(spec.shape), got); err != nil {
				t.Fatal(err)
			}
			for i, v := range got {
				if v != fill {
					t.Fatalf("[%d] want fill %d got %d", i, fill, v)
				}
			}
		})
	})
	t.Run("float32", func(t *testing.T) {
		rapid.Check(t, func(t *rapid.T) {
			spec := drawSpec(t)
			fill := rapid.Float32Range(-1e6, 1e6).Draw(t, "fill")
			s := propWriteArray(t, spec.shape, spec.chunks, "float32", nil, fill)
			a, err := zarr.OpenArray[float32](context.Background(), s, "arr")
			if err != nil {
				t.Fatal(err)
			}
			n := totalElems(spec.shape)
			got := make([]float32, n)
			if err := a.Read(context.Background(), zarr.All(spec.shape), got); err != nil {
				t.Fatal(err)
			}
			for i, v := range got {
				if v != fill {
					t.Fatalf("[%d] want fill %v got %v", i, fill, v)
				}
			}
		})
	})
	t.Run("float64", func(t *testing.T) {
		rapid.Check(t, func(t *rapid.T) {
			spec := drawSpec(t)
			fill := rapid.Float64Range(-1e9, 1e9).Draw(t, "fill")
			s := propWriteArray(t, spec.shape, spec.chunks, "float64", nil, fill)
			a, err := zarr.OpenArray[float64](context.Background(), s, "arr")
			if err != nil {
				t.Fatal(err)
			}
			n := totalElems(spec.shape)
			got := make([]float64, n)
			if err := a.Read(context.Background(), zarr.All(spec.shape), got); err != nil {
				t.Fatal(err)
			}
			for i, v := range got {
				if v != fill {
					t.Fatalf("[%d] want fill %v got %v", i, fill, v)
				}
			}
		})
	})
	t.Run("uint8", func(t *testing.T) {
		rapid.Check(t, func(t *rapid.T) {
			spec := drawSpec(t)
			fill := rapid.Uint8().Draw(t, "fill")
			s := propWriteArray(t, spec.shape, spec.chunks, "uint8", nil, fill)
			a, err := zarr.OpenArray[uint8](context.Background(), s, "arr")
			if err != nil {
				t.Fatal(err)
			}
			n := totalElems(spec.shape)
			got := make([]uint8, n)
			if err := a.Read(context.Background(), zarr.All(spec.shape), got); err != nil {
				t.Fatal(err)
			}
			for i, v := range got {
				if v != fill {
					t.Fatalf("[%d] want fill %d got %d", i, fill, v)
				}
			}
		})
	})
}
