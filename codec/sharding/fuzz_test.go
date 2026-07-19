package sharding

import (
	"encoding/json"
	"testing"

	_ "github.com/rkm/zarr-go/codec/bytes"
	_ "github.com/rkm/zarr-go/codec/crc32c"
)

// FuzzDecodeIndex feeds arbitrary bytes to decodeIndex, the parser for a
// shard's (offset, length) index trailer. Those bytes come directly from
// the store — untrusted input, since a corrupted or hostile shard blob is
// attacker-influenced data, not just internal state — so decodeIndex must
// never panic on malformed input: it must either return a valid n-entry
// table or a clean error.
func FuzzDecodeIndex(f *testing.F) {
	c := &Codec{}
	cfg, _ := json.Marshal(map[string]any{
		"chunk_shape": []int{2, 2},
		"codecs": []map[string]any{
			{"name": "bytes", "configuration": map[string]any{"endian": "little"}},
		},
	})
	if err := c.Configure(cfg); err != nil {
		f.Fatalf("Configure: %v", err)
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
			// Bounds a corpus entry can grow the entry count to during
			// fuzzing; decodeIndex itself is safe for any n (allocation is
			// gated by a length check against the actual input first — see
			// sharding.go's decodeIndex/DecodeAB), but there's nothing more
			// to learn from huge n once that fast-reject path is exercised.
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
