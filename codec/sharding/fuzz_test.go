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

// FuzzDecodeIndex feeds arbitrary bytes to decodeIndex, the parser for a
// shard's (offset, length) index trailer. Those bytes come directly from
// the store — untrusted input, since a corrupted or hostile shard blob is
// attacker-influenced data, not just internal state — so decodeIndex must
// never panic on malformed input: it must either return a valid n-entry
// table or a clean error.
func FuzzDecodeIndex(f *testing.F) {
	cfg, _ := json.Marshal(map[string]any{
		"chunk_shape": []int{2, 2},
		"codecs": []map[string]any{
			{"name": "bytes", "configuration": map[string]any{"endian": "little"}},
		},
	})
	c, err := New(cfg)
	if err != nil {
		f.Fatalf("New: %v", err)
	}

	// Seed with real encoded indexes (default index_codecs: bytes+crc32c)
	// for a few small n, so the fuzzer starts from valid input and can
	// mutate byte-for-byte (flipped bytes, truncation, extension) from
	// there.
	for _, n := range []int64{0, 1, 4} {
		entries := make([]uint64, n*2)
		for i := range entries {
			entries[i] = uint64(i) * 12345
		}
		encoded, err := c.encodeIndexEntries(entries)
		if err != nil {
			f.Fatalf("encodeIndexEntries(n=%d): %v", n, err)
		}
		f.Add(encoded, n)
	}

	f.Fuzz(func(t *testing.T, data []byte, n int64) {
		if n < 0 || n > 1<<16 {
			// decodeIndex is safe for any n, but huge n only exercises the
			// same fast-reject path, so keep the fuzzer on small n.
			return
		}
		entries, err := c.decodeIndex(data, n)
		if err != nil {
			return
		}
		if int64(len(entries)) != n*2 {
			t.Fatalf("decodeIndex(n=%d) returned %d entries, want %d", n, len(entries), n*2)
		}
	})
}

// FuzzDecodePartial feeds arbitrary shard bytes and region bounds to
// DecodePartial, served from a mem store. The shard blob, and so its
// index, is untrusted store data; Array.Read runs DecodePartial in a
// goroutine, so it must return an error rather than panic.
func FuzzDecodePartial(f *testing.F) {
	const dim = 8
	// Index without crc32c so mutated index entries reach the range logic
	// instead of failing the checksum.
	cfg, _ := json.Marshal(map[string]any{
		"chunk_shape": []int{2},
		"codecs": []map[string]any{
			{"name": "bytes", "configuration": map[string]any{"endian": "little"}},
		},
		"index_codecs": []map[string]any{
			{"name": "bytes", "configuration": map[string]any{"endian": "little"}},
		},
	})
	c, err := New(cfg)
	if err != nil {
		f.Fatalf("New: %v", err)
	}

	valid, err := c.EncodeAB(codec.ArrayChunk{
		Shape: []int64{dim}, Dtype: "uint8",
		Data: []byte{1, 2, 0, 0, 5, 6, 7, 8}, // inner chunk 1 absent
	})
	if err != nil {
		f.Fatalf("EncodeAB: %v", err)
	}
	// Index entry whose off+len overflows int64.
	malicious := []byte{1, 2, 3, 4}
	for _, e := range []uint64{0, 4, 1000, math.MaxInt64, 0, 2, 2, 2} {
		malicious = binary.LittleEndian.AppendUint64(malicious, e)
	}
	for _, shard := range [][]byte{valid, malicious} {
		f.Add(shard, int64(0), int64(dim))
		f.Add(shard, int64(1), int64(5))
	}

	f.Fuzz(func(t *testing.T, shard []byte, start, stop int64) {
		st := mem.New()
		if err := st.Set(context.Background(), "shard", shard); err != nil {
			t.Fatal(err)
		}
		var out []byte
		if 0 <= start && start <= stop && stop <= dim {
			out = make([]byte, stop-start)
		}
		req := codec.PartialReadRequest{
			Key: "shard", FullShape: []int64{dim}, Dtype: "uint8",
			Region: []codec.DimRange{{Start: start, Stop: stop, Step: 1}},
			Out:    out, GapBytes: 4 << 20,
		}
		_ = c.DecodePartial(context.Background(), st, &req)
	})
}
