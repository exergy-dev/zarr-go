package zlib

import (
	"encoding/json"
	"math/rand"
	"testing"

	"github.com/exergy-dev/zarr-go/internal/declimit"
)

func TestZlibRoundTrip(t *testing.T) {
	c := &Codec{}
	if err := c.Configure(json.RawMessage(`{"level":5}`)); err != nil {
		t.Fatal(err)
	}
	cases := [][]byte{
		[]byte("hello world"),
		make([]byte, 0),
		randBytes(t, 1<<10),
		randBytes(t, 1<<16),
	}
	for _, in := range cases {
		enc, err := c.EncodeBB(in)
		if err != nil {
			t.Fatal(err)
		}
		dec, err := c.DecodeBB(enc)
		if err != nil {
			t.Fatal(err)
		}
		if string(dec) != string(in) {
			t.Fatalf("round trip mismatch: len in=%d out=%d", len(in), len(dec))
		}
	}
}

func TestZlibConfigureLevels(t *testing.T) {
	c := &Codec{}
	for _, raw := range []string{`{}`, `{"level":-1}`, `{"level":0}`, `{"level":9}`} {
		if err := c.Configure(json.RawMessage(raw)); err != nil {
			t.Fatalf("config %q: %v", raw, err)
		}
	}
	for _, raw := range []string{`{"level":10}`, `{"level":-2}`, `{`} {
		if err := c.Configure(json.RawMessage(raw)); err == nil {
			t.Fatalf("config %q: expected error", raw)
		}
	}
}

func TestZlibDecodeGarbage(t *testing.T) {
	c := &Codec{}
	_ = c.Configure(nil)
	if _, err := c.DecodeBB([]byte("not zlib")); err == nil {
		t.Fatal("expected error decoding garbage")
	}
}

// TestDecodeBombBoundedByDeclimit locks in the decompression-bomb defense:
// a highly compressible payload that expands well past a lowered cap must
// be rejected with a bounded error, never fully materialized in memory.
func TestDecodeBombBoundedByDeclimit(t *testing.T) {
	orig := declimit.Max()
	defer declimit.SetMax(orig)

	c := &Codec{}
	if err := c.Configure(json.RawMessage(`{"level":5}`)); err != nil {
		t.Fatal(err)
	}
	huge := make([]byte, 8<<20) // 8 MiB of zeros compresses to a tiny stream
	enc, err := c.EncodeBB(huge)
	if err != nil {
		t.Fatalf("EncodeBB: %v", err)
	}
	if len(enc) >= len(huge)/100 {
		t.Fatalf("encoded size = %d, want a large reduction from %d", len(enc), len(huge))
	}

	declimit.SetMax(1024) // far below the real decompressed size
	if _, err := c.DecodeBB(enc); err == nil {
		t.Fatal("expected decode to be rejected once the decompressed size exceeds the cap, got nil")
	}
}

func randBytes(t *testing.T, n int) []byte {
	t.Helper()
	r := rand.New(rand.NewSource(42))
	b := make([]byte, n)
	r.Read(b)
	return b
}
