package zlib

import (
	"encoding/json"
	"math/rand"
	"testing"
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

func randBytes(t *testing.T, n int) []byte {
	t.Helper()
	r := rand.New(rand.NewSource(42))
	b := make([]byte, n)
	r.Read(b)
	return b
}
