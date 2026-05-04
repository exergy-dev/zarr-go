package gzip

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"testing"
)

// newConfigured returns a fresh Codec configured from the given JSON. An
// empty raw value selects the codec's default level.
func newConfigured(t *testing.T, raw string) *Codec {
	t.Helper()
	c := &Codec{}
	var msg json.RawMessage
	if raw != "" {
		msg = json.RawMessage(raw)
	}
	if err := c.Configure(msg); err != nil {
		t.Fatalf("Configure(%q): %v", raw, err)
	}
	return c
}

func TestRoundTripSmall(t *testing.T) {
	c := newConfigured(t, "")
	in := []byte("hello, zarr v3 gzip codec!")
	enc, err := c.EncodeBB(in)
	if err != nil {
		t.Fatalf("EncodeBB: %v", err)
	}
	dec, err := c.DecodeBB(enc)
	if err != nil {
		t.Fatalf("DecodeBB: %v", err)
	}
	if !bytes.Equal(in, dec) {
		t.Fatalf("round-trip mismatch:\n got %q\nwant %q", dec, in)
	}
}

func TestRoundTripLargeRandom(t *testing.T) {
	c := newConfigured(t, "")
	const n = 1 << 20 // 1 MiB
	in := make([]byte, n)
	r := rand.New(rand.NewSource(0xC0FFEE))
	if _, err := r.Read(in); err != nil {
		t.Fatalf("rand.Read: %v", err)
	}
	enc, err := c.EncodeBB(in)
	if err != nil {
		t.Fatalf("EncodeBB: %v", err)
	}
	dec, err := c.DecodeBB(enc)
	if err != nil {
		t.Fatalf("DecodeBB: %v", err)
	}
	if !bytes.Equal(in, dec) {
		t.Fatalf("round-trip mismatch on %d-byte buffer", n)
	}
}

func TestConfigureLevels(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want int
	}{
		{"default-empty", "", 5},
		{"default-empty-obj", `{}`, 5},
		{"explicit-default", `{"level":-1}`, -1},
		{"level-0", `{"level":0}`, 0},
		{"level-5", `{"level":5}`, 5},
		{"level-9", `{"level":9}`, 9},
	}

	in := bytes.Repeat([]byte("the quick brown fox jumps over the lazy dog\n"), 256)

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := newConfigured(t, tc.raw)
			if c.Level != tc.want {
				t.Fatalf("Level = %d, want %d", c.Level, tc.want)
			}
			enc, err := c.EncodeBB(in)
			if err != nil {
				t.Fatalf("EncodeBB: %v", err)
			}
			dec, err := c.DecodeBB(enc)
			if err != nil {
				t.Fatalf("DecodeBB: %v", err)
			}
			if !bytes.Equal(in, dec) {
				t.Fatalf("round-trip mismatch at level %d", c.Level)
			}
		})
	}
}

func TestConfigureInvalid(t *testing.T) {
	bad := []string{
		`{"level":10}`,
		`{"level":-2}`,
		`not-json`,
	}
	for _, raw := range bad {
		c := &Codec{}
		if err := c.Configure(json.RawMessage(raw)); err == nil {
			t.Fatalf("Configure(%q): expected error, got nil", raw)
		}
	}
}

func TestDecodeGarbage(t *testing.T) {
	c := newConfigured(t, "")
	// Garbage that is not a valid gzip stream (no 1f 8b magic).
	garbage := []byte("this is definitely not a gzip stream, no magic here")

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("DecodeBB panicked on garbage input: %v", r)
		}
	}()

	if _, err := c.DecodeBB(garbage); err == nil {
		t.Fatalf("DecodeBB(garbage): expected error, got nil")
	}
}

func TestDecodeTruncated(t *testing.T) {
	c := newConfigured(t, "")
	in := bytes.Repeat([]byte("payload"), 1024)
	enc, err := c.EncodeBB(in)
	if err != nil {
		t.Fatalf("EncodeBB: %v", err)
	}
	// Lop off the trailing checksum/length to force a stream error.
	truncated := enc[:len(enc)-4]

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("DecodeBB panicked on truncated input: %v", r)
		}
	}()

	if _, err := c.DecodeBB(truncated); err == nil {
		t.Fatalf("DecodeBB(truncated): expected error, got nil")
	}
}

func TestNameAndRegistration(t *testing.T) {
	c := &Codec{}
	if got := c.Name(); got != "gzip" {
		t.Fatalf("Name() = %q, want %q", got, "gzip")
	}
}
