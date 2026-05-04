package zarr_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/rkm/zarr-go"
	"github.com/rkm/zarr-go/codec"
	"github.com/rkm/zarr-go/store/mem"
)

// TestSharded_PartialReadEngine verifies the engine's partial-decode wiring:
// a sharded array is created, populated, then a sub-region read is issued.
// The result must match a reference whole-array read.
func TestSharded_PartialReadEngine(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	shape := []int64{16, 16}
	chunks := []int64{8, 8}
	a, err := zarr.CreateArray[int32](ctx, s, "arr", shape, chunks,
		zarr.WithCodecs(
			codec.Spec{Name: "sharding_indexed", Configuration: json.RawMessage(`{
				"chunk_shape": [2, 2],
				"codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
				"index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}]
			}`)},
		))
	if err != nil {
		t.Fatalf("CreateArray: %v", err)
	}
	src := make([]int32, shape[0]*shape[1])
	for i := range src {
		src[i] = int32(i + 1)
	}
	if err := a.Write(ctx, zarr.All(shape), src); err != nil {
		t.Fatalf("Write: %v", err)
	}

	// Reference: read the whole array.
	ref := make([]int32, shape[0]*shape[1])
	b, err := zarr.OpenArray[int32](ctx, s, "arr")
	if err != nil {
		t.Fatalf("OpenArray: %v", err)
	}
	if err := b.Read(ctx, zarr.All(shape), ref); err != nil {
		t.Fatalf("ref Read: %v", err)
	}
	for i := range src {
		if ref[i] != src[i] {
			t.Fatalf("ref[%d]: got %d want %d", i, ref[i], src[i])
		}
	}

	// Partial: rows [3,11) cols [4,12) — straddles chunk boundary at row 8 / col 8,
	// and within each shard chunk only intersects a few inner chunks.
	sel := zarr.Selection{Ranges: []zarr.DimRange{
		{Start: 3, Stop: 11, Step: 1},
		{Start: 4, Stop: 12, Step: 1},
	}}
	got := make([]int32, sel.NumElements())
	if err := b.Read(ctx, sel, got); err != nil {
		t.Fatalf("partial Read: %v", err)
	}
	// Compare against expected slice from ref.
	idx := 0
	for r := int64(3); r < 11; r++ {
		for c := int64(4); c < 12; c++ {
			want := ref[r*shape[1]+c]
			if got[idx] != want {
				t.Fatalf("[%d,%d] got %d want %d", r, c, got[idx], want)
			}
			idx++
		}
	}
}

// TestSharded_StridedReadEngine verifies that stepped selections still work
// (they fall back to the full-fetch path inside the engine).
func TestSharded_StridedReadEngine(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	shape := []int64{8, 8}
	chunks := []int64{8, 8}
	a, err := zarr.CreateArray[int32](ctx, s, "arr", shape, chunks,
		zarr.WithCodecs(
			codec.Spec{Name: "sharding_indexed", Configuration: json.RawMessage(`{
				"chunk_shape": [2, 2],
				"codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
				"index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}]
			}`)},
		))
	if err != nil {
		t.Fatalf("CreateArray: %v", err)
	}
	src := make([]int32, shape[0]*shape[1])
	for i := range src {
		src[i] = int32(i + 1)
	}
	if err := a.Write(ctx, zarr.All(shape), src); err != nil {
		t.Fatalf("Write: %v", err)
	}
	b, err := zarr.OpenArray[int32](ctx, s, "arr")
	if err != nil {
		t.Fatalf("OpenArray: %v", err)
	}

	// Stride 2 along both dims.
	sel := zarr.Selection{Ranges: []zarr.DimRange{
		{Start: 0, Stop: 8, Step: 2},
		{Start: 0, Stop: 8, Step: 2},
	}}
	got := make([]int32, sel.NumElements())
	if err := b.Read(ctx, sel, got); err != nil {
		t.Fatalf("strided Read: %v", err)
	}
	idx := 0
	for r := int64(0); r < 8; r += 2 {
		for c := int64(0); c < 8; c += 2 {
			want := src[r*shape[1]+c]
			if got[idx] != want {
				t.Fatalf("[%d,%d] got %d want %d", r, c, got[idx], want)
			}
			idx++
		}
	}
}
