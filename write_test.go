package zarr_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/rkm/zarr-go"
	"github.com/rkm/zarr-go/codec"
	"github.com/rkm/zarr-go/store/mem"
)

func TestWriteThenRead_Bytes_2D(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	shape := []int64{8, 8}
	chunks := []int64{4, 4}
	a, err := zarr.CreateArray[float32](ctx, s, "arr", shape, chunks)
	if err != nil {
		t.Fatal(err)
	}
	src := make([]float32, shape[0]*shape[1])
	for i := range src {
		src[i] = float32(i) * 0.5
	}
	if err := a.Write(ctx, zarr.All(shape), src); err != nil {
		t.Fatal(err)
	}
	// Re-open and read back.
	b, err := zarr.OpenArray[float32](ctx, s, "arr")
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, len(src))
	if err := b.Read(ctx, zarr.All(shape), got); err != nil {
		t.Fatal(err)
	}
	for i := range src {
		if got[i] != src[i] {
			t.Fatalf("[%d] want %v got %v", i, src[i], got[i])
		}
	}
}

func TestWriteThenRead_Gzip_3D(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	shape := []int64{4, 6, 8}
	chunks := []int64{2, 3, 4}
	a, err := zarr.CreateArray[int32](ctx, s, "arr", shape, chunks,
		zarr.WithCodecs(
			codec.Spec{Name: "bytes", Configuration: json.RawMessage(`{"endian":"little"}`)},
			codec.Spec{Name: "gzip", Configuration: json.RawMessage(`{"level":5}`)},
		))
	if err != nil {
		t.Fatal(err)
	}
	src := make([]int32, shape[0]*shape[1]*shape[2])
	for i := range src {
		src[i] = int32(i*17 - 100)
	}
	if err := a.Write(ctx, zarr.All(shape), src); err != nil {
		t.Fatal(err)
	}
	b, err := zarr.OpenArray[int32](ctx, s, "arr")
	if err != nil {
		t.Fatal(err)
	}
	got := make([]int32, len(src))
	if err := b.Read(ctx, zarr.All(shape), got); err != nil {
		t.Fatal(err)
	}
	for i := range src {
		if got[i] != src[i] {
			t.Fatalf("[%d] want %v got %v", i, src[i], got[i])
		}
	}
}

func TestWriteSliceReadModifyWrite(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	shape := []int64{8}
	chunks := []int64{4}
	a, err := zarr.CreateArray[int32](ctx, s, "arr", shape, chunks)
	if err != nil {
		t.Fatal(err)
	}
	// Fill the array with 1..8.
	full := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	if err := a.Write(ctx, zarr.All(shape), full); err != nil {
		t.Fatal(err)
	}
	// Overwrite [3..6] with 99,99,99.
	patch := []int32{99, 99, 99}
	if err := a.Write(ctx, zarr.Selection{Ranges: []zarr.DimRange{{3, 6, 1}}}, patch); err != nil {
		t.Fatal(err)
	}
	b, err := zarr.OpenArray[int32](ctx, s, "arr")
	if err != nil {
		t.Fatal(err)
	}
	got := make([]int32, 8)
	if err := b.Read(ctx, zarr.All(shape), got); err != nil {
		t.Fatal(err)
	}
	want := []int32{1, 2, 3, 99, 99, 99, 7, 8}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("[%d] want %v got %v", i, want[i], got[i])
		}
	}
}

func TestWriteEdgeChunkPadsWithFill(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	// shape 10, chunks 4 → last chunk has 2 valid + 2 padded slots.
	shape := []int64{10}
	chunks := []int64{4}
	a, err := zarr.CreateArray[float32](ctx, s, "arr", shape, chunks,
		zarr.WithFillValue(json.RawMessage("-1")))
	if err != nil {
		t.Fatal(err)
	}
	full := []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	if err := a.Write(ctx, zarr.All(shape), full); err != nil {
		t.Fatal(err)
	}
	b, err := zarr.OpenArray[float32](ctx, s, "arr")
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, 10)
	if err := b.Read(ctx, zarr.All(shape), got); err != nil {
		t.Fatal(err)
	}
	for i := range full {
		if got[i] != full[i] {
			t.Fatalf("[%d] want %v got %v", i, full[i], got[i])
		}
	}
}

func TestCreateGroupAndChild(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	g, err := zarr.CreateGroup(ctx, s, "root", map[string]any{"name": "root"})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.CreateGroup(ctx, "child", nil); err != nil {
		t.Fatal(err)
	}
	// Re-open as a node.
	n, err := zarr.Open(ctx, s, "root/child")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := n.(*zarr.Group); !ok {
		t.Fatalf("want *Group, got %T", n)
	}
}

func TestCreateArrayRejectsBadShape(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	if _, err := zarr.CreateArray[float32](ctx, s, "arr", []int64{4}, []int64{0}); err == nil {
		t.Fatal("expected error for zero chunk dim")
	}
	if _, err := zarr.CreateArray[float32](ctx, s, "arr", []int64{4, 4}, []int64{4}); err == nil {
		t.Fatal("expected error for rank mismatch")
	}
}
