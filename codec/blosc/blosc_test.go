//go:build cgo && zarrcgo

package blosc

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"

	"github.com/rkm/zarr-go/codec"
)

func newConfigured(t *testing.T, raw string) *Codec {
	t.Helper()
	c := &Codec{}
	if err := c.Configure(json.RawMessage(raw)); err != nil {
		t.Fatalf("Configure(%q): %v", raw, err)
	}
	return c
}

// makeFloat32MiB builds a 1 MiB buffer of float32 values whose pattern is
// shuffle/bitshuffle-friendly (smooth ramp -> good compression).
func makeFloat32MiB() []byte {
	const n = (1 << 20) / 4 // 262144 floats == 1 MiB
	out := make([]byte, n*4)
	for i := 0; i < n; i++ {
		v := math.Float32bits(float32(i) * 0.001)
		binary.LittleEndian.PutUint32(out[i*4:], v)
	}
	return out
}

func TestRoundTrip1MiBFloat32_LZ4_BitShuffle(t *testing.T) {
	in := makeFloat32MiB()
	c := newConfigured(t, `{"cname":"lz4","clevel":5,"shuffle":"bitshuffle","typesize":4}`)
	enc, err := c.EncodeBB(in)
	if err != nil {
		t.Fatalf("EncodeBB: %v", err)
	}
	if len(enc) == 0 || len(enc) >= len(in) {
		t.Fatalf("encoded size = %d, want a non-trivial reduction from %d", len(enc), len(in))
	}
	dec, err := c.DecodeBB(enc)
	if err != nil {
		t.Fatalf("DecodeBB: %v", err)
	}
	if !bytes.Equal(dec, in) {
		t.Fatalf("round-trip mismatch: %d != %d bytes", len(dec), len(in))
	}
}

func TestCLevels(t *testing.T) {
	in := makeFloat32MiB()
	for _, lvl := range []int{1, 3, 5, 9} {
		raw := []byte(`{"cname":"lz4","clevel":` + itoa(lvl) + `,"shuffle":"shuffle","typesize":4}`)
		c := &Codec{}
		if err := c.Configure(raw); err != nil {
			t.Fatalf("Configure clevel=%d: %v", lvl, err)
		}
		enc, err := c.EncodeBB(in)
		if err != nil {
			t.Fatalf("EncodeBB clevel=%d: %v", lvl, err)
		}
		dec, err := c.DecodeBB(enc)
		if err != nil {
			t.Fatalf("DecodeBB clevel=%d: %v", lvl, err)
		}
		if !bytes.Equal(dec, in) {
			t.Fatalf("clevel=%d round-trip mismatch", lvl)
		}
	}
}

func TestRoundTripEmpty(t *testing.T) {
	c := newConfigured(t, `{}`)
	enc, err := c.EncodeBB(nil)
	if err != nil {
		t.Fatalf("EncodeBB nil: %v", err)
	}
	if len(enc) != 0 {
		t.Fatalf("EncodeBB(nil) = %d bytes, want 0", len(enc))
	}
	dec, err := c.DecodeBB(enc)
	if err != nil {
		t.Fatalf("DecodeBB empty: %v", err)
	}
	if len(dec) != 0 {
		t.Fatalf("DecodeBB(empty) = %d bytes, want 0", len(dec))
	}
}

func TestRoundTripSnappyNoshuffle(t *testing.T) {
	c := &Codec{}
	if err := c.Configure(json.RawMessage(`{"cname":"blosclz","clevel":3,"shuffle":"noshuffle","typesize":1}`)); err != nil {
		t.Fatalf("Configure: %v", err)
	}
	in := bytes.Repeat([]byte("zarr-go blosc roundtrip "), 4096)
	enc, err := c.EncodeBB(in)
	if err != nil {
		t.Fatalf("EncodeBB: %v", err)
	}
	dec, err := c.DecodeBB(enc)
	if err != nil {
		t.Fatalf("DecodeBB: %v", err)
	}
	if !bytes.Equal(dec, in) {
		t.Fatal("blosclz noshuffle round-trip mismatch")
	}
}

func TestNameAndRegistration(t *testing.T) {
	c := &Codec{}
	if got := c.Name(); got != "blosc" {
		t.Fatalf("Name() = %q, want %q", got, "blosc")
	}
	f := codec.Lookup("blosc")
	if f == nil {
		t.Fatal("blosc codec not registered")
	}
	if _, ok := f().(codec.BytesToBytesCodec); !ok {
		t.Fatal("registered blosc codec does not implement BytesToBytesCodec")
	}
}

func TestConfigureRejectsBadShuffle(t *testing.T) {
	c := &Codec{}
	if err := c.Configure(json.RawMessage(`{"cname":"lz4","shuffle":"sideways"}`)); err == nil {
		t.Fatal("expected error for unknown shuffle")
	}
}

func TestConfigureRejectsOutOfRangeCLevel(t *testing.T) {
	c := &Codec{}
	if err := c.Configure(json.RawMessage(`{"cname":"lz4","clevel":15}`)); err == nil {
		t.Fatal("expected error for clevel out of range")
	}
}

func TestDecodeGarbage(t *testing.T) {
	c := newConfigured(t, `{}`)
	garbage := []byte{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66}
	if _, err := c.DecodeBB(garbage); err == nil {
		t.Fatal("expected error decoding garbage, got nil")
	}
}

// tiny strconv-free int formatter so tests don't need extra imports.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
