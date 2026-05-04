//go:build cgo && zarrcgo

package zstd

import (
	"bytes"
	"encoding/binary"
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

func TestCgoRoundTripSmall(t *testing.T) {
	c := newConfigured(t, `{"level":5}`)
	roundTrip(t, c, []byte("hello, zarr-go zstd codec!"))
}

func TestCgoRoundTripEmpty(t *testing.T) {
	c := newConfigured(t, ``)
	roundTrip(t, c, []byte{})
}

func TestCgoRoundTrip1MiBRandom(t *testing.T) {
	c := newConfigured(t, `{"level":3}`)
	buf := make([]byte, 1<<20)
	r := rand.New(rand.NewSource(0x5eed))
	if _, err := r.Read(buf); err != nil {
		t.Fatalf("rand read: %v", err)
	}
	roundTrip(t, c, buf)
}

func TestCgoRoundTripChecksum(t *testing.T) {
	c := newConfigured(t, `{"level":1,"checksum":true}`)
	if !c.Checksum {
		t.Fatal("expected Checksum=true after Configure")
	}
	payload := bytes.Repeat([]byte("zarr"), 1024)
	roundTrip(t, c, payload)
}

func TestCgoDecodeGarbage(t *testing.T) {
	c := newConfigured(t, ``)
	garbage := []byte{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0xff, 0xee}
	if _, err := c.DecodeBB(garbage); err == nil {
		t.Fatal("expected error decoding garbage, got nil")
	}
}

func TestCgoNameAndRegistration(t *testing.T) {
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

func TestCgoConfigureInvalidJSON(t *testing.T) {
	c := &Codec{}
	if err := c.Configure(json.RawMessage(`{not json`)); err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestCgoConfigureDefaults(t *testing.T) {
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

// TestCgoFrameContentSize verifies the encoded zstd frame carries the
// Frame_Content_Size (FCS) field in the header. The frame header layout
// (RFC 8478 §3.1.1.1) starts with the 4-byte Magic_Number 0xFD2FB528
// (little-endian) followed by Frame_Header_Descriptor; FCS_Flag is bits
// 6-7 of that descriptor.
func TestCgoFrameContentSize(t *testing.T) {
	c := newConfigured(t, `{"level":1}`)
	payload := bytes.Repeat([]byte("zarr-go-cgo-zstd"), 256)
	enc, err := c.EncodeBB(payload)
	if err != nil {
		t.Fatalf("EncodeBB: %v", err)
	}
	if len(enc) < 6 {
		t.Fatalf("encoded frame too short: %d bytes", len(enc))
	}
	magic := binary.LittleEndian.Uint32(enc[:4])
	if magic != 0xFD2FB528 {
		t.Fatalf("magic = %#x, want 0xFD2FB528", magic)
	}
	fhd := enc[4]
	fcsFlag := (fhd >> 6) & 0x3
	if fcsFlag == 0 {
		// FCS_Flag == 0 means: FCS field is 0 or 1 byte depending on
		// Single_Segment_Flag (bit 5). When the segment is single
		// (and small), FCS is exactly 1 byte. We accept that too.
		ssFlag := (fhd >> 5) & 0x1
		if ssFlag == 0 {
			t.Fatalf("frame header descriptor %#x has FCS_Flag=0 and Single_Segment_Flag=0 (no FCS)", fhd)
		}
	}
}
