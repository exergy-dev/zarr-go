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

// shortDecodeCodec is a deliberately buggy array->bytes codec whose EncodeAB
// is a correct identity passthrough but whose DecodeAB truncates the output.
// Used to prove decodeAndCopy rejects a mismatched decode length with a
// DecodeError instead of panicking inside copyChunkRegion's slicing.
type shortDecodeCodec struct{}

func (*shortDecodeCodec) Name() string                    { return "test_short_decode" }
func (*shortDecodeCodec) Configure(json.RawMessage) error { return nil }
func (*shortDecodeCodec) EncodeAB(in codec.ArrayChunk) ([]byte, error) {
	return append([]byte(nil), in.Data...), nil
}
func (*shortDecodeCodec) DecodeAB(in []byte, out *codec.ArrayChunk) error {
	out.Data = in[:len(in)/2]
	return nil
}

func init() {
	codec.Register("test_short_decode", func() codec.Codec { return &shortDecodeCodec{} })
}

// TestStreamEdgeChunkShapeMatchesData guards against Stream yielding a Chunk
// whose Data is laid out for the full (unclipped) chunk shape while Shape
// reports the clipped edge extent — a consumer indexing Data by Shape's
// strides would silently read the wrong element.
func TestStreamEdgeChunkShapeMatchesData(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	// 5x5 array, 4x4 chunks: chunk (0,1) has actual shape [4,1].
	a, err := zarr.CreateArray[int32](ctx, s, "arr", []int64{5, 5}, []int64{4, 4})
	if err != nil {
		t.Fatal(err)
	}
	src := make([]int32, 25)
	for i := range src {
		src[i] = int32(i)
	}
	if err := a.Write(ctx, zarr.All([]int64{5, 5}), src); err != nil {
		t.Fatal(err)
	}

	found := false
	for ck, err := range a.Stream(ctx, zarr.All([]int64{5, 5})) {
		if err != nil {
			t.Fatal(err)
		}
		n := int64(1)
		for _, d := range ck.Shape {
			n *= d
		}
		if int64(len(ck.Data)) != n {
			t.Fatalf("chunk %v: len(Data)=%d does not match Shape=%v (product=%d)",
				ck.Coord, len(ck.Data), ck.Shape, n)
		}
		if ck.Coord[0] == 0 && ck.Coord[1] == 1 {
			found = true
			// Global element (row=1, col=4) = value 9. Chunk-local (1,0).
			want := int32(9)
			got := ck.Data[1*ck.Shape[1]+0]
			if got != want {
				t.Fatalf("chunk (0,1) local (1,0): want %d got %d", want, got)
			}
		}
	}
	if !found {
		t.Fatal("expected to see edge chunk (0,1)")
	}
}

// TestReadCancelledContextReturnsError guards against Read returning nil with
// a partially-filled dst when the context is cancelled mid-flight.
func TestReadCancelledContextReturnsError(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	a, err := zarr.CreateArray[int32](ctx, s, "arr", []int64{64}, []int64{4})
	if err != nil {
		t.Fatal(err)
	}
	src := make([]int32, 64)
	for i := range src {
		src[i] = int32(i + 1) // non-zero so "still zero" implies "not filled"
	}
	if err := a.Write(ctx, zarr.All([]int64{64}), src); err != nil {
		t.Fatal(err)
	}

	for trial := 0; trial < 200; trial++ {
		cctx, cancel := context.WithCancel(ctx)
		cancel()
		dst := make([]int32, 64)
		err := a.Read(cctx, zarr.All([]int64{64}), dst)
		if err == nil {
			t.Fatalf("trial %d: Read with pre-cancelled context returned nil error", trial)
		}
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("trial %d: err = %v, want context.Canceled", trial, err)
		}
	}
}

// TestWriteEmptySelectionNoPanic guards against Write panicking on a valid
// zero-element selection (Start == Stop along some dim).
func TestWriteEmptySelectionNoPanic(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	a, err := zarr.CreateArray[float64](ctx, s, "arr", []int64{10}, []int64{4})
	if err != nil {
		t.Fatal(err)
	}
	sel := zarr.Selection{Ranges: []zarr.DimRange{{Start: 3, Stop: 3, Step: 1}}}
	if err := a.Write(ctx, sel, []float64{}); err != nil {
		t.Fatalf("Write(empty selection) returned error: %v", err)
	}
}

// TestDecodeShortChunkReturnsError guards against a codec that decodes to
// fewer bytes than the chunk's declared shape*dtype size causing a
// slice-bounds panic in copyChunkRegion instead of a clean DecodeError.
func TestDecodeShortChunkReturnsError(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	a, err := zarr.CreateArray[int32](ctx, s, "arr", []int64{8}, []int64{8},
		zarr.WithCodecs(codec.Spec{Name: "test_short_decode"}))
	if err != nil {
		t.Fatal(err)
	}
	src := make([]int32, 8)
	for i := range src {
		src[i] = int32(i + 1)
	}
	if err := a.Write(ctx, zarr.All([]int64{8}), src); err != nil {
		t.Fatal(err)
	}

	dst := make([]int32, 8)
	err = a.Read(ctx, zarr.All([]int64{8}), dst)
	if err == nil {
		t.Fatal("expected a decode error, got nil")
	}
	var de *zarr.DecodeError
	if !errors.As(err, &de) {
		t.Fatalf("expected *zarr.DecodeError, got %T: %v", err, err)
	}
}
