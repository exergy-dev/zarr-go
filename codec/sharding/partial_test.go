package sharding_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"sync/atomic"
	"testing"

	"github.com/exergy-dev/zarr-go/codec"
	_ "github.com/exergy-dev/zarr-go/codec/bytes"
	_ "github.com/exergy-dev/zarr-go/codec/crc32c"
	_ "github.com/exergy-dev/zarr-go/codec/gzip"
	"github.com/exergy-dev/zarr-go/store/mem"
)

// countingFetcher is a codec.RangeFetcher serving range and suffix reads
// from an in-memory blob, counting the calls of each kind.
type countingFetcher struct {
	key  string
	blob []byte
	gets int64 // GetRange calls
	suff int64 // GetSuffix calls
}

func (f *countingFetcher) GetRange(_ context.Context, key string, off, n int64) ([]byte, error) {
	atomic.AddInt64(&f.gets, 1)
	if key != f.key {
		return nil, errKey(key)
	}
	if off < 0 || off > int64(len(f.blob)) {
		return nil, errBounds(off)
	}
	end := off + n
	if end > int64(len(f.blob)) {
		end = int64(len(f.blob))
	}
	out := make([]byte, end-off)
	copy(out, f.blob[off:end])
	return out, nil
}

func (f *countingFetcher) GetSuffix(_ context.Context, key string, n int64) ([]byte, error) {
	atomic.AddInt64(&f.suff, 1)
	if key != f.key {
		return nil, errKey(key)
	}
	off := int64(len(f.blob)) - n
	if off < 0 {
		off = 0
		n = int64(len(f.blob))
	}
	out := make([]byte, n)
	copy(out, f.blob[off:off+n])
	return out, nil
}

type errKey string

func (e errKey) Error() string { return "unexpected key: " + string(e) }

type errBounds int64

func (e errBounds) Error() string { return "offset out of bounds" }

const cfgPartial = `{
  "chunk_shape": [2, 2],
  "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
  "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}]
}`

// TestDecodePartialFetchCount builds an 8x8 int32 shard with 4x4=16 inner
// chunks (chunk_shape=[2,2]). It then issues a partial decode for the
// rectangle rows[2:6), cols[2:6), which intersects exactly 4 inner chunks
// (rows 1-2 of the inner grid, cols 1-2). It verifies:
//
//  1. The bytes returned match a reference full-decode.
//  2. The fetcher saw at most 2 calls: one suffix-read for the index
//     trailer and one coalesced GetRange for the 4 contiguous inner chunks.
func TestDecodePartialFetchCount(t *testing.T) {
	c := newCodec(t, cfgPartial)
	pd, ok := c.(codec.PartialDecoder)
	if !ok {
		t.Fatalf("sharding_indexed does not implement PartialDecoder")
	}

	const dim = 8
	vals := make([]int32, dim*dim)
	for i := range vals {
		vals[i] = int32(i + 1) // distinct non-zero values
	}
	in := codec.ArrayChunk{
		Shape: []int64{dim, dim},
		Dtype: "int32",
		Data:  int32Bytes(vals),
	}
	encoded, err := c.EncodeAB(in)
	if err != nil {
		t.Fatalf("EncodeAB: %v", err)
	}

	// Reference: whole-shard decode.
	ref := codec.ArrayChunk{Shape: []int64{dim, dim}, Dtype: "int32"}
	if err := c.DecodeAB(encoded, &ref); err != nil {
		t.Fatalf("reference DecodeAB: %v", err)
	}

	// Partial decode: rows [2,6) cols [2,6) -> 4x4 = 16 elements.
	region := []codec.DimRange{
		{Start: 2, Stop: 6, Step: 1},
		{Start: 2, Stop: 6, Step: 1},
	}
	regionBytes := 4 * 4 * 4
	out := make([]byte, regionBytes)
	f := &countingFetcher{key: "shard", blob: encoded}
	req := codec.PartialReadRequest{
		Key: "shard", FullShape: []int64{dim, dim}, Dtype: "int32",
		Region: region, Out: out, GapBytes: 4 << 20,
	}
	if err := pd.DecodePartial(context.Background(), f, &req); err != nil {
		t.Fatalf("DecodePartial: %v", err)
	}

	// Build the expected slice from the reference whole-decode by extracting
	// rows 2..5, cols 2..5 from the row-major 8x8 buffer.
	want := make([]byte, regionBytes)
	for r := int64(0); r < 4; r++ {
		srcRow := (2+r)*dim + 2
		dstRow := r * 4
		copy(
			want[dstRow*4:dstRow*4+4*4],
			ref.Data[srcRow*4:srcRow*4+4*4],
		)
	}
	if !bytes.Equal(out, want) {
		t.Fatalf("partial decode mismatch\n got %v\nwant %v", bytesAsInt32(out), bytesAsInt32(want))
	}

	gets := atomic.LoadInt64(&f.gets)
	suff := atomic.LoadInt64(&f.suff)
	total := gets + suff
	t.Logf("fetch counts: GetRange=%d GetSuffix=%d total=%d", gets, suff, total)
	if total > 2 {
		t.Errorf("expected at most 2 fetcher calls (1 suffix + 1 coalesced), got %d (GetRange=%d, GetSuffix=%d)",
			total, gets, suff)
	}
	if suff != 1 {
		t.Errorf("expected exactly 1 GetSuffix call (index trailer), got %d", suff)
	}
	if gets != 1 {
		t.Errorf("expected exactly 1 GetRange call (coalesced inner chunks), got %d", gets)
	}
}

// TestDecodePartialEdgeCases exercises corner cases of the partial decoder
// (region equal to the full chunk, region of zero size, and a region that
// includes a fill (sentinel) inner chunk).
func TestDecodePartialEdgeCases(t *testing.T) {
	c := newCodec(t, cfgPartial)
	pd := c.(codec.PartialDecoder)

	const dim = 4
	// Build a 4x4 shard where inner chunk (0,1) (top-right 2x2) is all zero.
	vals := []int32{
		1, 2, 0, 0,
		5, 6, 0, 0,
		9, 10, 11, 12,
		13, 14, 15, 16,
	}
	encoded, err := c.EncodeAB(codec.ArrayChunk{
		Shape: []int64{dim, dim}, Dtype: "int32", Data: int32Bytes(vals),
	})
	if err != nil {
		t.Fatalf("EncodeAB: %v", err)
	}
	f := &countingFetcher{key: "shard", blob: encoded}

	// Case 1: region is the entire shard.
	out := make([]byte, dim*dim*4)
	region := []codec.DimRange{{Start: 0, Stop: dim, Step: 1}, {Start: 0, Stop: dim, Step: 1}}
	req := codec.PartialReadRequest{
		Key: "shard", FullShape: []int64{dim, dim}, Dtype: "int32",
		Region: region, Out: out, GapBytes: 4 << 20,
	}
	if err := pd.DecodePartial(context.Background(), f, &req); err != nil {
		t.Fatalf("full-region DecodePartial: %v", err)
	}
	if !bytes.Equal(out, int32Bytes(vals)) {
		t.Errorf("full-region DecodePartial mismatch")
	}

	// Case 2: region intersects only the empty inner chunk -> all zeros.
	out2 := make([]byte, 2*2*4)
	region2 := []codec.DimRange{{Start: 0, Stop: 2, Step: 1}, {Start: 2, Stop: 4, Step: 1}}
	req2 := codec.PartialReadRequest{
		Key: "shard", FullShape: []int64{dim, dim}, Dtype: "int32",
		Region: region2, Out: out2, GapBytes: 4 << 20,
	}
	if err := pd.DecodePartial(context.Background(), f, &req2); err != nil {
		t.Fatalf("empty-region DecodePartial: %v", err)
	}
	for _, b := range out2 {
		if b != 0 {
			t.Errorf("expected zero region from empty inner chunk, got %v", bytesAsInt32(out2))
			break
		}
	}
}

// cfgRawIndex has no crc32c on the index, so tests can hand-write index
// entries without recomputing a checksum.
const cfgRawIndex = `{
  "chunk_shape": [2],
  "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
  "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
}`

// rawShard lays out data followed by a little-endian uint64 index trailer.
func rawShard(data []byte, entries ...uint64) []byte {
	out := append([]byte(nil), data...)
	for _, e := range entries {
		out = binary.LittleEndian.AppendUint64(out, e)
	}
	return out
}

// TestDecodePartialRejectsBadIndexEntries feeds hostile (offset, length)
// index entries through a mem store. Each must produce an error, never a panic:
// Array.Read runs DecodePartial in a goroutine, so a panic here takes down
// the process.
func TestDecodePartialRejectsBadIndexEntries(t *testing.T) {
	const empty = math.MaxUint64
	data := []byte{1, 2, 3, 4}
	cases := []struct {
		name    string
		entries []uint64 // (off, len) for inner chunks 0 and 1
	}{
		// off+len wraps int64: must not be treated as a range fully
		// contained in an earlier one (blob[start:start+N] would panic).
		{"overflow", []uint64{0, 4, 1000, math.MaxInt64}},
		{"overflow-first", []uint64{1, math.MaxInt64, 0, 2}},
		{"half-empty-off", []uint64{0, 2, empty, 2}},
		{"half-empty-len", []uint64{0, 2, 2, empty}},
		{"past-end", []uint64{0, 2, 30, 10}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			st := mem.New()
			if err := st.Set(context.Background(), "shard", rawShard(data, tc.entries...)); err != nil {
				t.Fatal(err)
			}
			pd := newCodec(t, cfgRawIndex).(codec.PartialDecoder)
			req := codec.PartialReadRequest{
				Key: "shard", FullShape: []int64{4}, Dtype: "uint8",
				Region: []codec.DimRange{{Start: 0, Stop: 3, Step: 1}},
				Out:    make([]byte, 3), GapBytes: 4 << 20,
			}
			if err := pd.DecodePartial(context.Background(), st, &req); err == nil {
				t.Fatalf("DecodePartial accepted index %v", tc.entries)
			}
		})
	}
}

// shortAB is a test-only array->bytes codec whose DecodeAB ignores out.Shape
// and returns half the input, so a sharding inner pipeline built on it yields
// inner chunks of the wrong size. No in-tree AB codec does this.
type shortAB struct{}

func (shortAB) Name() string                                 { return "sharding_test_short_ab" }
func (shortAB) EncodeAB(in codec.ArrayChunk) ([]byte, error) { return in.Data, nil }
func (shortAB) DecodeAB(in []byte, out *codec.ArrayChunk) error {
	out.Data = in[:len(in)/2]
	return nil
}

func init() {
	codec.Register("sharding_test_short_ab", func(json.RawMessage) (codec.Codec, error) {
		return shortAB{}, nil
	})
}

// TestDecodePartialRejectsWrongInnerSize: an inner pipeline that decodes to
// the wrong number of bytes must be reported as an error (as DecodeAB does),
// not copied out of bounds.
func TestDecodePartialRejectsWrongInnerSize(t *testing.T) {
	const cfg = `{
  "chunk_shape": [4],
  "codecs": [{"name": "sharding_test_short_ab"}],
  "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
}`
	shard := rawShard([]byte{1, 2, 3, 4}, 0, 4)
	pd := newCodec(t, cfg).(codec.PartialDecoder)
	req := codec.PartialReadRequest{
		Key: "shard", FullShape: []int64{4}, Dtype: "uint8",
		Region: []codec.DimRange{{Start: 0, Stop: 4, Step: 1}},
		Out:    make([]byte, 4), GapBytes: 4 << 20,
	}
	f := &countingFetcher{key: "shard", blob: shard}
	if err := pd.DecodePartial(context.Background(), f, &req); err == nil {
		t.Fatalf("DecodePartial accepted a 2-byte inner chunk for shape [4] uint8, out=%v", req.Out)
	}
}

// Compile-time interface assertion.
var _ codec.RangeFetcher = (*countingFetcher)(nil)
