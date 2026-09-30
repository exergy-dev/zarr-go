package zstd

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math/rand"
	"testing"

	"github.com/exergy-dev/zarr-go/codec"
	"github.com/exergy-dev/zarr-go/internal/declimit"
)

func newConfigured(t *testing.T, raw string) *Codec {
	t.Helper()
	c, err := New(json.RawMessage(raw))
	if err != nil {
		t.Fatalf("New(%q) failed: %v", raw, err)
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

// frameHasChecksum reports the Content_Checksum_flag (bit 2 of the
// Frame_Header_Descriptor) of a zstd frame; see RFC 8478 §3.1.1.1.1.
func frameHasChecksum(t *testing.T, frame []byte) bool {
	t.Helper()
	if len(frame) < 5 || binary.LittleEndian.Uint32(frame[:4]) != 0xFD2FB528 {
		t.Fatalf("not a zstd frame: % x", frame[:min(len(frame), 8)])
	}
	return frame[4]&0x04 != 0
}

func TestRoundTripChecksum(t *testing.T) {
	c := newConfigured(t, `{"level":1,"checksum":true}`)
	payload := bytes.Repeat([]byte("zarr"), 1024)
	roundTrip(t, c, payload)
	enc, err := c.EncodeBB(payload)
	if err != nil {
		t.Fatalf("EncodeBB: %v", err)
	}
	if !frameHasChecksum(t, enc) {
		t.Fatal("checksum:true produced a frame without a content checksum")
	}
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
	inst, err := f(nil)
	if err != nil {
		t.Fatalf("factory(nil): %v", err)
	}
	if _, ok := inst.(codec.BytesToBytesCodec); !ok {
		t.Fatalf("registered zstd codec does not implement BytesToBytesCodec (got %T)", inst)
	}
}

func TestNewInvalid(t *testing.T) {
	for _, raw := range []string{`{not json`, `{"level":"fast"}`, `{"checksum":1}`} {
		c, err := New(json.RawMessage(raw))
		if err == nil {
			t.Fatalf("New(%q): expected error", raw)
		}
		if c != nil {
			t.Fatalf("New(%q): returned a codec alongside the error", raw)
		}
	}
}

// TestDecodeBombRejectsForgedFrameContentSize crafts a zstd frame whose
// header declares a huge Frame_Content_Size but whose actual compressed
// body is tiny, simulating a decompression bomb. DecodeBB must reject it
// based on the declared size alone, without attempting to allocate anywhere
// near the forged size.
func TestDecodeBombRejectsForgedFrameContentSize(t *testing.T) {
	orig := declimit.Max()
	defer declimit.SetMax(orig)
	declimit.SetMax(1 << 30) // default cap; forged size below is well above it

	c := newConfigured(t, ``)
	// A payload large enough (>65791 bytes) that WithSingleSegment(true)
	// picks the 4-byte Frame_Content_Size field (FCS_Flag=2).
	payload := bytes.Repeat([]byte{0}, 100000)
	enc, err := c.EncodeBB(payload)
	if err != nil {
		t.Fatalf("EncodeBB: %v", err)
	}
	forged := forgeFrameContentSize(t, enc, 0xF0000000) // ~4 GiB

	c2 := newConfigured(t, ``) // fresh instance: New samples the cap
	if _, err := c2.DecodeBB(forged); err == nil {
		t.Fatal("expected error decoding a frame with a forged oversized Frame_Content_Size, got nil")
	}
}

// forgeFrameContentSize patches the little-endian Frame_Content_Size field
// of a single-segment zstd frame (as produced by this codec's EncodeBB,
// which always sets WithSingleSegment(true)) to v, leaving the rest of the
// frame — including the compressed body — untouched. Requires the frame's
// FCS_Flag to select a 4-byte field (payload size in [65792, 1<<32)); see
// RFC 8478 §3.1.1.
func forgeFrameContentSize(t *testing.T, frame []byte, v uint32) []byte {
	t.Helper()
	if len(frame) < 9 {
		t.Fatalf("frame too short: %d bytes", len(frame))
	}
	magic := binary.LittleEndian.Uint32(frame[:4])
	if magic != 0xFD2FB528 {
		t.Fatalf("magic = %#x, want 0xFD2FB528", magic)
	}
	fhd := frame[4]
	singleSegment := (fhd>>5)&1 == 1
	fcsFlag := (fhd >> 6) & 0x3
	dictIDFlag := fhd & 0x3
	if !singleSegment || fcsFlag != 2 || dictIDFlag != 0 {
		t.Fatalf("frame header descriptor %#x: want single-segment, FCS_Flag=2, no dict ID; adjust payload size", fhd)
	}
	out := append([]byte(nil), frame...)
	binary.LittleEndian.PutUint32(out[5:9], v)
	return out
}

// TestNewDefaults checks that an absent or empty configuration builds a
// codec that omits the content checksum.
func TestNewDefaults(t *testing.T) {
	for _, raw := range []string{``, `{}`, `{"checksum":false}`} {
		c := newConfigured(t, raw)
		enc, err := c.EncodeBB([]byte("zarr-go zstd defaults"))
		if err != nil {
			t.Fatalf("EncodeBB: %v", err)
		}
		if frameHasChecksum(t, enc) {
			t.Errorf("config %q: frame has a content checksum, want none", raw)
		}
	}
}
