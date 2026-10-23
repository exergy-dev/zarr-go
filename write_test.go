package zarr_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/codec"
	"github.com/exergy-dev/zarr-go/store/mem"
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
	if err := a.Write(ctx, zarr.Selection{{Start: 3, Stop: 6, Step: 1}}, patch); err != nil {
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
	g, err := zarr.CreateGroup(ctx, s, "root", zarr.WithAttributes(map[string]any{"name": "root"}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.CreateGroup(ctx, "child"); err != nil {
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

// selFlatIndices returns, in the selection's row-major output order, the
// flat row-major index into an array of the given shape of every element
// sel covers: an independent reference for what Read/Write touch.
func selFlatIndices(shape []int64, sel zarr.Selection) []int64 {
	out := []int64{0}
	for d, r := range sel {
		var next []int64
		for _, base := range out {
			for i := r.Start; i < r.Stop; i += r.Step {
				next = append(next, base*shape[d]+i)
			}
		}
		out = next
	}
	return out
}

// TestReadErrorAmongManyChunks checks that one corrupt chunk among many,
// read with a single I/O and decode worker, surfaces as a DecodeError
// naming that chunk rather than hanging or being lost.
func TestReadErrorAmongManyChunks(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	shape := []int64{64}
	a, err := zarr.CreateArray[int32](ctx, s, "arr", shape, []int64{1},
		zarr.WithCodecs(
			codec.Spec{Name: "bytes", Configuration: json.RawMessage(`{"endian":"little"}`)},
			codec.Spec{Name: "gzip", Configuration: json.RawMessage(`{"level":1}`)},
		),
		zarr.WithConcurrency(1), zarr.WithIOConcurrency(1))
	if err != nil {
		t.Fatalf("CreateArray: %v", err)
	}
	if err := a.Write(ctx, zarr.All(shape), make([]int32, 64)); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err := s.Set(ctx, "arr/c/37", []byte("corrupt")); err != nil {
		t.Fatal(err)
	}
	err = a.Read(ctx, zarr.All(shape), make([]int32, 64))
	var de *zarr.DecodeError
	if !errors.As(err, &de) {
		t.Fatalf("Read: got %v, want *zarr.DecodeError", err)
	}
	if len(de.Coord) != 1 || de.Coord[0] != 37 {
		t.Errorf("DecodeError.Coord = %v, want [37]", de.Coord)
	}
}

// TestWriteReadHighRankStrided round-trips strided writes and reads through
// a rank-9 array with hundreds of chunks, many edge-clipped, under tight
// concurrency limits (more chunks than workers), and checks every element
// against an in-memory reference.
func TestWriteReadHighRankStrided(t *testing.T) {
	ctx := context.Background()
	const rank = 9
	shape := make([]int64, rank)
	chunks := make([]int64, rank)
	for i := range shape {
		shape[i], chunks[i] = 3, 2
	}
	a, err := zarr.CreateArray[int32](ctx, mem.New(), "arr", shape, chunks,
		zarr.WithConcurrency(2), zarr.WithIOConcurrency(3))
	if err != nil {
		t.Fatalf("CreateArray: %v", err)
	}
	full := zarr.All(shape)
	ref := make([]int32, full.NumElements())
	for i := range ref {
		ref[i] = int32(i + 1)
	}
	if err := a.Write(ctx, full, ref); err != nil {
		t.Fatalf("Write(all): %v", err)
	}

	strided := make([]zarr.DimRange, rank)
	for i := range strided {
		strided[i] = zarr.DimRange{Start: int64(i % 2), Stop: 3, Step: 2}
	}
	sel := zarr.Selection(strided)
	patch := make([]int32, sel.NumElements())
	for i, flat := range selFlatIndices(shape, sel) {
		patch[i] = -int32(i + 1)
		ref[flat] = patch[i]
	}
	if err := a.Write(ctx, sel, patch); err != nil {
		t.Fatalf("Write(strided): %v", err)
	}

	got := make([]int32, len(ref))
	if err := a.Read(ctx, full, got); err != nil {
		t.Fatalf("Read(all): %v", err)
	}
	for i := range ref {
		if got[i] != ref[i] {
			t.Fatalf("Read(all)[%d] = %d, want %d", i, got[i], ref[i])
		}
	}

	for i := range strided {
		strided[i] = zarr.DimRange{Start: int64(i % 3), Stop: 3, Step: 1 + int64(i%2)}
	}
	sel = zarr.Selection(strided)
	got = make([]int32, sel.NumElements())
	if err := a.Read(ctx, sel, got); err != nil {
		t.Fatalf("Read(strided): %v", err)
	}
	for i, flat := range selFlatIndices(shape, sel) {
		if got[i] != ref[flat] {
			t.Fatalf("Read(strided)[%d] = %d, want %d", i, got[i], ref[flat])
		}
	}
}
