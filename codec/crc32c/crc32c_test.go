package crc32c

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"math/rand"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	c := &Codec{}
	sizes := []int{0, 1, 7, 64, 1024, 1 << 20}
	for _, n := range sizes {
		n := n
		t.Run("", func(t *testing.T) {
			in := make([]byte, n)
			r := rand.New(rand.NewSource(int64(n) + 1))
			r.Read(in)

			enc, err := c.EncodeBB(in)
			if err != nil {
				t.Fatalf("EncodeBB(%d): %v", n, err)
			}
			if len(enc) != n+4 {
				t.Fatalf("encoded length = %d, want %d", len(enc), n+4)
			}
			// Trailer must be little-endian CRC32C of body.
			wantSum := crc32.Checksum(in, crc32.MakeTable(crc32.Castagnoli))
			gotSum := binary.LittleEndian.Uint32(enc[n:])
			if wantSum != gotSum {
				t.Fatalf("trailer = %x, want %x", gotSum, wantSum)
			}

			dec, err := c.DecodeBB(enc)
			if err != nil {
				t.Fatalf("DecodeBB: %v", err)
			}
			if !bytes.Equal(dec, in) {
				t.Fatalf("round-trip mismatch (n=%d)", n)
			}
		})
	}
}

func TestEncodeReturnsFreshSlice(t *testing.T) {
	c := &Codec{}
	in := []byte{1, 2, 3, 4, 5}
	enc, err := c.EncodeBB(in)
	if err != nil {
		t.Fatal(err)
	}
	// Mutating in should not affect enc.
	in[0] = 0xff
	if enc[0] != 1 {
		t.Fatalf("EncodeBB shared backing array with input")
	}
}

func TestTamperedBody(t *testing.T) {
	c := &Codec{}
	in := []byte("the quick brown fox jumps over the lazy dog")
	enc, err := c.EncodeBB(in)
	if err != nil {
		t.Fatal(err)
	}
	// Flip a bit in the body.
	enc[3] ^= 0x01
	if _, err := c.DecodeBB(enc); err == nil {
		t.Fatal("expected error from tampered body, got nil")
	}
}

func TestTamperedTrailer(t *testing.T) {
	c := &Codec{}
	in := []byte("the quick brown fox jumps over the lazy dog")
	enc, err := c.EncodeBB(in)
	if err != nil {
		t.Fatal(err)
	}
	// Flip a bit in the trailer (last 4 bytes).
	enc[len(enc)-1] ^= 0x80
	if _, err := c.DecodeBB(enc); err == nil {
		t.Fatal("expected error from tampered trailer, got nil")
	}
}

func TestTruncatedInput(t *testing.T) {
	c := &Codec{}
	for _, n := range []int{0, 1, 2, 3} {
		n := n
		t.Run("", func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("DecodeBB panicked on len=%d input: %v", n, r)
				}
			}()
			_, err := c.DecodeBB(make([]byte, n))
			if err == nil {
				t.Fatalf("expected error for len=%d input, got nil", n)
			}
		})
	}
}

func TestDecodeReturnsFreshSlice(t *testing.T) {
	c := &Codec{}
	in := []byte{10, 20, 30, 40, 50, 60}
	enc, err := c.EncodeBB(in)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := c.DecodeBB(enc)
	if err != nil {
		t.Fatal(err)
	}
	// Mutating enc should not affect dec.
	enc[0] = 0xff
	if dec[0] != 10 {
		t.Fatal("DecodeBB shared backing array with input")
	}
}

func TestConfigure(t *testing.T) {
	c := &Codec{}
	cases := [][]byte{
		nil,
		[]byte(""),
		[]byte("{}"),
		[]byte(`{"unused":true}`),
	}
	for _, raw := range cases {
		if err := c.Configure(raw); err != nil {
			t.Fatalf("Configure(%q): %v", raw, err)
		}
	}
	if err := c.Configure([]byte("not json")); err == nil {
		t.Fatal("expected error on malformed JSON")
	}
}

func TestName(t *testing.T) {
	if (&Codec{}).Name() != "crc32c" {
		t.Fatalf("Name() = %q, want %q", (&Codec{}).Name(), "crc32c")
	}
}
