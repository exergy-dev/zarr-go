package sharding

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"

	"github.com/exergy-dev/zarr-go/codec"
	_ "github.com/exergy-dev/zarr-go/codec/bytes"
	_ "github.com/exergy-dev/zarr-go/codec/crc32c"
	"github.com/exergy-dev/zarr-go/store/mem"
)

// TestEncodeDecodeFillNonZero confirms that an inner chunk whose elements
// equal the array's fill_value is stored as the empty-marker sentinel and
// reads back as fill_value.
func TestEncodeDecodeFillNonZero(t *testing.T) {
	// 4x4 outer shard, 2x2 inner chunks → 4 inner chunks. fill = NaN.
	cfg, _ := json.Marshal(map[string]any{
		"chunk_shape": []int{2, 2},
		"codecs": []map[string]any{
			{"name": "bytes", "configuration": map[string]any{"endian": "little"}},
		},
	})
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Build a 4x4 float32 buffer where one inner chunk is all-NaN (fill),
	// one is all-zero, two contain real data. The all-NaN chunk MUST round-
	// trip back to NaN, and the all-zero chunk MUST round-trip back to 0.
	const elem = 4
	data := make([]byte, 4*4*elem)
	put := func(r, col int, v float32) {
		binary.LittleEndian.PutUint32(data[(r*4+col)*elem:], math.Float32bits(v))
	}
	for r := 0; r < 4; r++ {
		for col := 0; col < 4; col++ {
			put(r, col, 0)
		}
	}
	// Top-left 2x2 block = NaN (matches fill).
	for r := 0; r < 2; r++ {
		for col := 0; col < 2; col++ {
			put(r, col, float32(math.NaN()))
		}
	}
	// Top-right 2x2 block = 1,2,3,4 (real data).
	put(0, 2, 1)
	put(0, 3, 2)
	put(1, 2, 3)
	put(1, 3, 4)
	// Bottom-right 2x2 block = -1 (real data, no zero/NaN match).
	for r := 2; r < 4; r++ {
		for col := 2; col < 4; col++ {
			put(r, col, -1)
		}
	}
	// Bottom-left 2x2 block stays all-zero (test that fill != 0 keeps it
	// stored, decoded as 0).

	// fill = NaN bytes (one element worth, native-LE float32).
	fill := make([]byte, elem)
	binary.LittleEndian.PutUint32(fill, math.Float32bits(float32(math.NaN())))

	in := codec.ArrayChunk{
		Shape: []int64{4, 4},
		Dtype: "float32",
		Data:  data,
		Fill:  fill,
	}
	encoded, err := c.EncodeAB(in)
	if err != nil {
		t.Fatalf("EncodeAB: %v", err)
	}

	out := codec.ArrayChunk{
		Shape: []int64{4, 4},
		Dtype: "float32",
		Fill:  fill,
	}
	if err := c.DecodeAB(encoded, &out); err != nil {
		t.Fatalf("DecodeAB: %v", err)
	}
	if len(out.Data) != len(data) {
		t.Fatalf("decoded size %d != original %d", len(out.Data), len(data))
	}
	get := func(r, col int) float32 {
		return math.Float32frombits(binary.LittleEndian.Uint32(out.Data[(r*4+col)*elem:]))
	}
	if !math.IsNaN(float64(get(0, 0))) {
		t.Errorf("top-left[0,0] = %v, want NaN", get(0, 0))
	}
	if !math.IsNaN(float64(get(1, 1))) {
		t.Errorf("top-left[1,1] = %v, want NaN", get(1, 1))
	}
	if get(0, 2) != 1 || get(1, 3) != 4 {
		t.Errorf("top-right corrupt: %v %v", get(0, 2), get(1, 3))
	}
	if get(2, 0) != 0 || get(3, 1) != 0 {
		t.Errorf("bottom-left non-zero: %v %v", get(2, 0), get(3, 1))
	}
	if get(2, 2) != -1 || get(3, 3) != -1 {
		t.Errorf("bottom-right corrupt: %v %v", get(2, 2), get(3, 3))
	}
}

// TestEncodeDecodeIndexStart exercises the index_location=start shard
// layout end-to-end.
func TestEncodeDecodeIndexStart(t *testing.T) {
	cfg, _ := json.Marshal(map[string]any{
		"chunk_shape":    []int{2, 2},
		"index_location": "start",
		"codecs": []map[string]any{
			{"name": "bytes", "configuration": map[string]any{"endian": "little"}},
		},
	})
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if c.indexLocation != indexStart {
		t.Fatalf("indexLocation = %v, want indexStart", c.indexLocation)
	}

	// 4x4 int32 with all-distinct values so any miscopy shows up.
	const elem = 4
	data := make([]byte, 4*4*elem)
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(data[i*elem:], uint32(i+1))
	}
	in := codec.ArrayChunk{
		Shape: []int64{4, 4},
		Dtype: "int32",
		Data:  data,
	}
	encoded, err := c.EncodeAB(in)
	if err != nil {
		t.Fatalf("EncodeAB: %v", err)
	}
	out := codec.ArrayChunk{
		Shape: []int64{4, 4},
		Dtype: "int32",
	}
	if err := c.DecodeAB(encoded, &out); err != nil {
		t.Fatalf("DecodeAB: %v", err)
	}
	for i := 0; i < 16; i++ {
		got := binary.LittleEndian.Uint32(out.Data[i*elem:])
		if got != uint32(i+1) {
			t.Fatalf("idx-start round-trip elem %d = %d, want %d", i, got, i+1)
		}
	}
}

// TestDecodePartialRespectsFill exercises the partial-decode path with a
// non-zero fill and an absent inner chunk; the corresponding outBuf region
// must come back as fill, not zero.
func TestDecodePartialRespectsFill(t *testing.T) {
	cfg, _ := json.Marshal(map[string]any{
		"chunk_shape": []int{2, 2},
		"codecs": []map[string]any{
			{"name": "bytes", "configuration": map[string]any{"endian": "little"}},
		},
	})
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Build a 4x4 shard with the top-left inner chunk all-fill (sentinel)
	// and the rest = 1,2,...
	const elem = 4
	data := make([]byte, 4*4*elem)
	for i := 0; i < 16; i++ {
		binary.LittleEndian.PutUint32(data[i*elem:], uint32(i+1))
	}
	// Overwrite top-left 2x2 with the fill value (sentinel pattern).
	const fillVal uint32 = 0xCAFEBABE
	fill := make([]byte, elem)
	binary.LittleEndian.PutUint32(fill, fillVal)
	for r := 0; r < 2; r++ {
		for col := 0; col < 2; col++ {
			binary.LittleEndian.PutUint32(data[(r*4+col)*elem:], fillVal)
		}
	}
	encoded, err := c.EncodeAB(codec.ArrayChunk{
		Shape: []int64{4, 4},
		Dtype: "uint32",
		Data:  data,
		Fill:  fill,
	})
	if err != nil {
		t.Fatalf("EncodeAB: %v", err)
	}

	// Serve the encoded blob from a mem store, partial decode the FULL outer
	// region and confirm fill bytes appear where we expect.
	f := mem.New()
	if err := f.Set(context.Background(), "shard", encoded); err != nil {
		t.Fatal(err)
	}
	region := []codec.DimRange{{Start: 0, Stop: 4, Step: 1}, {Start: 0, Stop: 4, Step: 1}}
	out := make([]byte, 4*4*elem)
	req := codec.PartialReadRequest{
		Key: "shard", FullShape: []int64{4, 4}, Dtype: "uint32",
		Region: region, Out: out, Fill: fill, GapBytes: 4 << 20,
	}
	if err := c.DecodePartial(context.Background(), f, &req); err != nil {
		t.Fatalf("DecodePartial: %v", err)
	}
	// Top-left must be fill bytes; the rest must be 1..16 except those four.
	for r := 0; r < 4; r++ {
		for col := 0; col < 4; col++ {
			got := binary.LittleEndian.Uint32(out[(r*4+col)*elem:])
			var want uint32
			if r < 2 && col < 2 {
				want = fillVal
			} else {
				want = uint32(r*4 + col + 1)
			}
			if got != want {
				t.Fatalf("[%d,%d] = %#x, want %#x", r, col, got, want)
			}
		}
	}
}
