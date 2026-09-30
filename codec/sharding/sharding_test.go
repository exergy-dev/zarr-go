package sharding_test

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"testing"

	"github.com/exergy-dev/zarr-go/codec"
	_ "github.com/exergy-dev/zarr-go/codec/bytes"
	_ "github.com/exergy-dev/zarr-go/codec/crc32c"
	_ "github.com/exergy-dev/zarr-go/codec/gzip"
	_ "github.com/exergy-dev/zarr-go/codec/sharding"
)

// newCodec builds a fresh sharding_indexed codec from JSON config.
func newCodec(t *testing.T, cfg string) codec.ArrayToBytesCodec {
	t.Helper()
	f := codec.Lookup("sharding_indexed")
	if f == nil {
		t.Fatal("sharding_indexed not registered")
	}
	c, err := f(json.RawMessage(cfg))
	if err != nil {
		t.Fatalf("factory: %v", err)
	}
	ab, ok := c.(codec.ArrayToBytesCodec)
	if !ok {
		t.Fatalf("not an ArrayToBytesCodec")
	}
	return ab
}

// int32Bytes returns the little-endian byte view of vals.
func int32Bytes(vals []int32) []byte {
	out := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], uint32(v))
	}
	return out
}

// bytesAsInt32 reverses int32Bytes.
func bytesAsInt32(b []byte) []int32 {
	out := make([]int32, len(b)/4)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

const cfgBytesOnly = `{
  "chunk_shape": [2, 2],
  "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
  "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}]
}`

const cfgBytesGzip = `{
  "chunk_shape": [2, 2],
  "codecs": [
    {"name": "bytes", "configuration": {"endian": "little"}},
    {"name": "gzip", "configuration": {"level": 5}}
  ],
  "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}]
}`

// TestRoundTripFull encodes a full 4x4 int32 shard with 2x2 inner chunks, then
// decodes and verifies bit-for-bit equality.
func TestRoundTripFull(t *testing.T) {
	c := newCodec(t, cfgBytesOnly)
	// 4x4 = 16 elements, distinct non-zero values.
	vals := []int32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	in := codec.ArrayChunk{
		Shape: []int64{4, 4},
		Dtype: "int32",
		Data:  int32Bytes(vals),
	}
	encoded, err := c.EncodeAB(in)
	if err != nil {
		t.Fatalf("EncodeAB: %v", err)
	}

	out := codec.ArrayChunk{Shape: []int64{4, 4}, Dtype: "int32"}
	if err := c.DecodeAB(encoded, &out); err != nil {
		t.Fatalf("DecodeAB: %v", err)
	}
	got := bytesAsInt32(out.Data)
	if len(got) != len(vals) {
		t.Fatalf("len mismatch: got %d, want %d", len(got), len(vals))
	}
	for i := range vals {
		if got[i] != vals[i] {
			t.Errorf("element %d: got %d, want %d", i, got[i], vals[i])
		}
	}
}

// TestRoundTripAllEmpty: zero-valued shard should round-trip and the index
// should mark every inner chunk as empty.
func TestRoundTripAllEmpty(t *testing.T) {
	c := newCodec(t, cfgBytesOnly)
	in := codec.ArrayChunk{
		Shape: []int64{4, 4},
		Dtype: "int32",
		Data:  make([]byte, 4*4*4),
	}
	encoded, err := c.EncodeAB(in)
	if err != nil {
		t.Fatalf("EncodeAB: %v", err)
	}
	// With 4 inner chunks, all empty: body is empty, only index trailer.
	// Index plain size = 4 * 16 = 64 bytes; +4 for crc32c.
	if len(encoded) != 68 {
		t.Errorf("expected encoded size 68 (all-empty index trailer), got %d", len(encoded))
	}
	out := codec.ArrayChunk{Shape: []int64{4, 4}, Dtype: "int32"}
	if err := c.DecodeAB(encoded, &out); err != nil {
		t.Fatalf("DecodeAB: %v", err)
	}
	if !bytes.Equal(out.Data, in.Data) {
		t.Errorf("decoded data not all zero")
	}
}

// TestRoundTripMixedEmpty has one inner chunk all zero, others non-zero.
// Verifies the index entry for the empty chunk uses the sentinel and decode
// fills it with zeros.
func TestRoundTripMixedEmpty(t *testing.T) {
	c := newCodec(t, cfgBytesOnly)
	// Inner chunk indexing for 4x4 shard with 2x2 chunks:
	//   inner 0 -> top-left (rows 0-1, cols 0-1)
	//   inner 1 -> top-right
	//   inner 2 -> bottom-left
	//   inner 3 -> bottom-right
	// Make inner chunk 1 (top-right) all zero.
	vals := []int32{
		1, 2, 0, 0,
		5, 6, 0, 0,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	in := codec.ArrayChunk{
		Shape: []int64{4, 4},
		Dtype: "int32",
		Data:  int32Bytes(vals),
	}
	encoded, err := c.EncodeAB(in)
	if err != nil {
		t.Fatalf("EncodeAB: %v", err)
	}
	out := codec.ArrayChunk{Shape: []int64{4, 4}, Dtype: "int32"}
	if err := c.DecodeAB(encoded, &out); err != nil {
		t.Fatalf("DecodeAB: %v", err)
	}
	got := bytesAsInt32(out.Data)
	for i := range vals {
		if got[i] != vals[i] {
			t.Errorf("element %d: got %d, want %d", i, got[i], vals[i])
		}
	}

	// Verify the index trailer marks inner chunk 1 as empty (sentinel).
	// 4 inner chunks * 16 bytes = 64; +4 crc32c = 68 trailer.
	idxStart := len(encoded) - 68
	idx := encoded[idxStart : idxStart+64]
	off1 := binary.LittleEndian.Uint64(idx[1*16:])
	len1 := binary.LittleEndian.Uint64(idx[1*16+8:])
	const max = ^uint64(0)
	if off1 != max || len1 != max {
		t.Errorf("inner chunk 1 should be empty sentinel, got off=%d len=%d", off1, len1)
	}
}

// TestRoundTripGzipInner exercises a non-trivial inner pipeline: bytes + gzip.
func TestRoundTripGzipInner(t *testing.T) {
	c := newCodec(t, cfgBytesGzip)
	vals := make([]int32, 16)
	for i := range vals {
		vals[i] = int32(100 + i)
	}
	in := codec.ArrayChunk{
		Shape: []int64{4, 4},
		Dtype: "int32",
		Data:  int32Bytes(vals),
	}
	encoded, err := c.EncodeAB(in)
	if err != nil {
		t.Fatalf("EncodeAB: %v", err)
	}
	out := codec.ArrayChunk{Shape: []int64{4, 4}, Dtype: "int32"}
	if err := c.DecodeAB(encoded, &out); err != nil {
		t.Fatalf("DecodeAB: %v", err)
	}
	got := bytesAsInt32(out.Data)
	for i := range vals {
		if got[i] != vals[i] {
			t.Errorf("element %d: got %d, want %d", i, got[i], vals[i])
		}
	}
}

// TestEdgeChunkClipping verifies inner chunks at the shard boundary are sized
// smaller than chunk_shape. Shard 5x3 with 2x2 chunk_shape yields a 3x2 grid
// where the last column and last row chunks are clipped.
func TestEdgeChunkClipping(t *testing.T) {
	c := newCodec(t, cfgBytesOnly)
	const rows, cols = 5, 3
	vals := make([]int32, rows*cols)
	for i := range vals {
		vals[i] = int32(i + 1)
	}
	in := codec.ArrayChunk{
		Shape: []int64{rows, cols},
		Dtype: "int32",
		Data:  int32Bytes(vals),
	}
	encoded, err := c.EncodeAB(in)
	if err != nil {
		t.Fatalf("EncodeAB: %v", err)
	}
	out := codec.ArrayChunk{Shape: []int64{rows, cols}, Dtype: "int32"}
	if err := c.DecodeAB(encoded, &out); err != nil {
		t.Fatalf("DecodeAB: %v", err)
	}
	got := bytesAsInt32(out.Data)
	for i := range vals {
		if got[i] != vals[i] {
			t.Errorf("element %d: got %d, want %d", i, got[i], vals[i])
		}
	}
}

// TestRegistered ensures init registered the codec.
func TestRegistered(t *testing.T) {
	if codec.Lookup("sharding_indexed") == nil {
		t.Fatal("sharding_indexed not registered")
	}
}
