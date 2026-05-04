//go:build !cgo || !zarrcgo

package zstd

import (
	"bytes"
	"encoding/json"
	"math/rand"
	"testing"

	"github.com/rkm/zarr-go/codec"
)

func newConfigured(t *testing.T, raw string) *Codec {
	t.Helper()
	c := &Codec{}
	if err := c.Configure(json.RawMessage(raw)); err != nil {
		t.Fatalf("Configure(%q) failed: %v", raw, err)
	}
	return c
}

func roundTrip(t *testing.T, c *Codec, in []byte) {
	t.Helper()
	enc, err := c.EncodeBB(in)
	if err != nil {
		t.Fatalf("EncodeBB error: %v", err)
	}
	dec, err := c.DecodeBB(enc)
	if err != nil {
		t.Fatalf("DecodeBB error: %v", err)
	}
	if !bytes.Equal(dec, in) {
		t.Fatalf("round-trip mismatch: got %d bytes, want %d", len(dec), len(in))
	}
}

func TestRoundTripSmall(t *testing.T) {
	c := newConfigured(t, `{"level":5}`)
	roundTrip(t, c, []byte("hello, zarr-go zstd codec!"))
}

func TestRoundTripEmpty(t *testing.T) {
	c := newConfigured(t, ``)
	roundTrip(t, c, []byte{})
}

func TestRoundTrip1MiBRandom(t *testing.T) {
	c := newConfigured(t, `{"level":3}`)
	buf := make([]byte, 1<<20)
	r := rand.New(rand.NewSource(0x5eed))
	if _, err := r.Read(buf); err != nil {
		t.Fatalf("rand read: %v", err)
	}
	roundTrip(t, c, buf)
}

func TestRoundTripChecksum(t *testing.T) {
	c := newConfigured(t, `{"level":1,"checksum":true}`)
	if !c.Checksum {
		t.Fatal("expected Checksum=true after Configure")
	}
	payload := bytes.Repeat([]byte("zarr"), 1024)
	roundTrip(t, c, payload)
}

func TestDecodeGarbage(t *testing.T) {
	c := newConfigured(t, ``)
	garbage := []byte{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0xff, 0xee}
	if _, err := c.DecodeBB(garbage); err == nil {
		t.Fatal("expected error decoding garbage, got nil")
	}
}

func TestNameAndRegistration(t *testing.T) {
	c := &Codec{}
	if got := c.Name(); got != "zstd" {
		t.Fatalf("Name() = %q, want %q", got, "zstd")
	}
	f := codec.Lookup("zstd")
	if f == nil {
		t.Fatal("zstd codec not registered")
	}
	inst := f()
	if _, ok := inst.(codec.BytesToBytesCodec); !ok {
		t.Fatalf("registered zstd codec does not implement BytesToBytesCodec (got %T)", inst)
	}
}

func TestConfigureInvalidJSON(t *testing.T) {
	c := &Codec{}
	if err := c.Configure(json.RawMessage(`{not json`)); err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestConfigureDefaults(t *testing.T) {
	c := &Codec{Level: 7, Checksum: true}
	if err := c.Configure(json.RawMessage(`{}`)); err != nil {
		t.Fatalf("Configure({}) failed: %v", err)
	}
	if c.Level != 0 {
		t.Errorf("Level = %d, want 0", c.Level)
	}
	if c.Checksum {
		t.Error("Checksum = true, want false")
	}
}
