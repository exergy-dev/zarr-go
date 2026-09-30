//go:build cgo && zarrcgo

package blosc

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"

	"github.com/exergy-dev/zarr-go/codec"
	"github.com/exergy-dev/zarr-go/internal/declimit"
)

func newConfigured(t *testing.T, raw string) *Codec {
	t.Helper()
	c, err := New(json.RawMessage(raw))
	if err != nil {
		t.Fatalf("New(%q): %v", raw, err)
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
		c, err := New(raw)
		if err != nil {
			t.Fatalf("New clevel=%d: %v", lvl, err)
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
	c := newConfigured(t, `{"cname":"blosclz","clevel":3,"shuffle":"noshuffle","typesize":1}`)
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
	inst, err := f(nil)
	if err != nil {
		t.Fatalf("factory(nil): %v", err)
	}
	if _, ok := inst.(codec.BytesToBytesCodec); !ok {
		t.Fatal("registered blosc codec does not implement BytesToBytesCodec")
	}
}

func TestNewRejectsBadShuffle(t *testing.T) {
	if _, err := New(json.RawMessage(`{"cname":"lz4","shuffle":"sideways"}`)); err == nil {
		t.Fatal("expected error for unknown shuffle")
	}
}

func TestNewRejectsOutOfRangeCLevel(t *testing.T) {
	if _, err := New(json.RawMessage(`{"cname":"lz4","clevel":15}`)); err == nil {
		t.Fatal("expected error for clevel out of range")
	}
}

func TestNewRejectsUnknownCompressor(t *testing.T) {
	if _, err := New(json.RawMessage(`{"cname":"no-such-compressor"}`)); err == nil {
		t.Fatal("expected error for a compressor libblosc lacks")
	}
}

func TestDecodeGarbage(t *testing.T) {
	c := newConfigured(t, `{}`)
	garbage := []byte{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66}
	if _, err := c.DecodeBB(garbage); err == nil {
		t.Fatal("expected error decoding garbage, got nil")
	}
}

// TestDecodeBombRejectsForgedNbytes crafts a blosc frame whose 16-byte
// header declares a huge uncompressed size (nbytes) but whose actual
// compressed body is tiny, simulating a decompression bomb. DecodeBB must
// reject it based on the declared nbytes alone, without attempting to
// allocate anywhere near the forged size (the pre-fix code did
// `make([]byte, int(nbytes))` straight from the untrusted header).
//
// c-blosc's 16-byte header layout: byte0=version, byte1=versionlz,
// byte2=flags, byte3=typesize, bytes[4:8]=nbytes (LE uint32),
// bytes[8:12]=blocksize (LE uint32), bytes[12:16]=cbytes (LE uint32). Only
// nbytes is forged; cbytes must still equal len(frame) or DecodeBB's own
// length-mismatch check (checked first) would mask the size-cap check.
func TestDecodeBombRejectsForgedNbytes(t *testing.T) {
	orig := declimit.Max()
	defer declimit.SetMax(orig)
	declimit.SetMax(1 << 30) // default cap; forged size below is well above it

	c := newConfigured(t, `{"cname":"lz4","clevel":5,"shuffle":"noshuffle","typesize":1}`)
	in := bytes.Repeat([]byte{0}, 4096)
	enc, err := c.EncodeBB(in)
	if err != nil {
		t.Fatalf("EncodeBB: %v", err)
	}
	if len(enc) < 16 {
		t.Fatalf("encoded frame too short: %d bytes", len(enc))
	}
	forged := append([]byte(nil), enc...)
	binary.LittleEndian.PutUint32(forged[4:8], 0xF0000000) // ~4 GiB declared nbytes

	if _, err := c.DecodeBB(forged); err == nil {
		t.Fatal("expected error decoding a frame with a forged oversized nbytes header, got nil")
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
