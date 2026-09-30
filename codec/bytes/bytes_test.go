package bytes

import (
	stdbytes "bytes"
	"encoding/json"
	"fmt"
	"testing"
	"unsafe"

	"github.com/exergy-dev/zarr-go/codec"
)

// dtypes lists every Zarr v3 dtype the bytes codec supports, paired with its
// element size in bytes.
var dtypes = []struct {
	name string
	size int
}{
	{"bool", 1},
	{"int8", 1},
	{"uint8", 1},
	{"int16", 2},
	{"uint16", 2},
	{"float16", 2},
	{"int32", 4},
	{"uint32", 4},
	{"float32", 4},
	{"int64", 8},
	{"uint64", 8},
	{"float64", 8},
	{"complex64", 8},
	{"complex128", 16},
}

// makeData fills a buffer with a deterministic non-trivial byte pattern so
// that endianness bugs would surface as different output bytes.
func makeData(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*31 + 7)
	}
	return b
}

func configure(t *testing.T, endian string) *Codec {
	t.Helper()
	c := &Codec{}
	cfg := json.RawMessage(fmt.Sprintf(`{"endian":%q}`, endian))
	if err := c.Configure(cfg); err != nil {
		t.Fatalf("Configure(%q): %v", endian, err)
	}
	return c
}

func TestRoundTripAllDtypes(t *testing.T) {
	for _, dt := range dtypes {
		for _, endian := range []string{"little", "big"} {
			t.Run(dt.name+"_"+endian, func(t *testing.T) {
				const nElems = 7
				orig := makeData(nElems * dt.size)
				in := codec.ArrayChunk{
					Shape: []int64{int64(nElems)},
					Dtype: dt.name,
					Data:  append([]byte(nil), orig...),
				}
				c := configure(t, endian)

				encoded, err := c.EncodeAB(in)
				if err != nil {
					t.Fatalf("EncodeAB: %v", err)
				}
				if len(encoded) != len(orig) {
					t.Fatalf("encoded len = %d, want %d", len(encoded), len(orig))
				}
				// EncodeAB must return a fresh slice — mutating it must not touch input.
				if len(encoded) > 0 && &encoded[0] == &in.Data[0] {
					t.Fatalf("EncodeAB returned a slice aliasing input")
				}

				out := codec.ArrayChunk{
					Shape: []int64{int64(nElems)},
					Dtype: dt.name,
				}
				if err := c.DecodeAB(encoded, &out); err != nil {
					t.Fatalf("DecodeAB: %v", err)
				}
				if !stdbytes.Equal(out.Data, orig) {
					t.Fatalf("round-trip mismatch:\n got  %x\n want %x", out.Data, orig)
				}
			})
		}
	}
}

// TestEncodeMatchesEndianness verifies that for multi-byte dtypes the encoded
// output is byte-identical to the input when configured for the host's native
// endianness, and byte-reversed (per element / per complex component) otherwise.
func TestEncodeMatchesEndianness(t *testing.T) {
	for _, dt := range dtypes {
		if dt.size == 1 {
			continue
		}
		const nElems = 5
		orig := makeData(nElems * dt.size)
		in := codec.ArrayChunk{
			Shape: []int64{int64(nElems)},
			Dtype: dt.name,
			Data:  append([]byte(nil), orig...),
		}

		nativeEndian := "big"
		if hostLittleEndian {
			nativeEndian = "little"
		}
		c := configure(t, nativeEndian)
		got, err := c.EncodeAB(in)
		if err != nil {
			t.Fatalf("%s: EncodeAB: %v", dt.name, err)
		}
		if !stdbytes.Equal(got, orig) {
			t.Fatalf("%s native (%s): expected memcpy, got swapped bytes", dt.name, nativeEndian)
		}

		other := "big"
		if nativeEndian == "big" {
			other = "little"
		}
		c2 := configure(t, other)
		got2, err := c2.EncodeAB(in)
		if err != nil {
			t.Fatalf("%s: EncodeAB(%s): %v", dt.name, other, err)
		}
		// Build the expected swap manually.
		unit := swapUnit(dt.name, dt.size)
		want := make([]byte, len(orig))
		for i := 0; i < len(orig); i += unit {
			for j := 0; j < unit; j++ {
				want[i+j] = orig[i+unit-1-j]
			}
		}
		if !stdbytes.Equal(got2, want) {
			t.Fatalf("%s non-native (%s): got %x want %x", dt.name, other, got2, want)
		}
	}
}

// TestFloat16LittleEndian explicitly checks the float16 little-endian path:
// the bit pattern 0x3C00 (== 1.0) must serialize to {0x00, 0x3C} on a LE host
// and to {0x3C, 0x00} when configured for big-endian.
func TestFloat16LittleEndian(t *testing.T) {
	// 1.0 in IEEE binary16 is 0x3C00.
	src := []byte{0x00, 0x3C, 0x00, 0x3C} // two elements, native LE host layout
	if !hostLittleEndian {
		// On a BE host the same logical value would be stored as 0x3C 0x00.
		src = []byte{0x3C, 0x00, 0x3C, 0x00}
	}
	in := codec.ArrayChunk{Shape: []int64{2}, Dtype: "float16", Data: src}

	cLE := configure(t, "little")
	encLE, err := cLE.EncodeAB(in)
	if err != nil {
		t.Fatalf("EncodeAB(little): %v", err)
	}
	wantLE := []byte{0x00, 0x3C, 0x00, 0x3C}
	if !stdbytes.Equal(encLE, wantLE) {
		t.Fatalf("float16 LE encoding: got %x want %x", encLE, wantLE)
	}

	cBE := configure(t, "big")
	encBE, err := cBE.EncodeAB(in)
	if err != nil {
		t.Fatalf("EncodeAB(big): %v", err)
	}
	wantBE := []byte{0x3C, 0x00, 0x3C, 0x00}
	if !stdbytes.Equal(encBE, wantBE) {
		t.Fatalf("float16 BE encoding: got %x want %x", encBE, wantBE)
	}

	// Round-trip both back.
	for _, tc := range []struct {
		name string
		c    *Codec
		enc  []byte
	}{{"little", cLE, encLE}, {"big", cBE, encBE}} {
		out := codec.ArrayChunk{Shape: []int64{2}, Dtype: "float16"}
		if err := tc.c.DecodeAB(tc.enc, &out); err != nil {
			t.Fatalf("DecodeAB(%s): %v", tc.name, err)
		}
		if !stdbytes.Equal(out.Data, src) {
			t.Fatalf("float16 round-trip (%s): got %x want %x", tc.name, out.Data, src)
		}
	}
}

func TestConfigureDefaults(t *testing.T) {
	c := &Codec{}
	if err := c.Configure(nil); err != nil {
		t.Fatalf("Configure(nil): %v", err)
	}
	if c.Endian != Little {
		t.Fatalf("default endian = %v, want Little", c.Endian)
	}

	c2 := &Codec{}
	if err := c2.Configure(json.RawMessage(`{}`)); err != nil {
		t.Fatalf("Configure({}): %v", err)
	}
	if c2.Endian != Little {
		t.Fatalf("empty-object endian = %v, want Little", c2.Endian)
	}
}

func TestConfigureInvalidEndian(t *testing.T) {
	for _, bad := range []string{"native", "LITTLE", "", "middle"} {
		c := &Codec{}
		raw := json.RawMessage(fmt.Sprintf(`{"endian":%q}`, bad))
		err := c.Configure(raw)
		if err == nil {
			t.Fatalf("Configure(endian=%q): expected error, got nil", bad)
		}
	}
}

func TestConfigureMalformedJSON(t *testing.T) {
	c := &Codec{}
	if err := c.Configure(json.RawMessage(`{not json`)); err == nil {
		t.Fatal("expected error for malformed JSON")
	}
}

func TestDecodeReusesBuffer(t *testing.T) {
	const nElems = 16
	dt := "float64"
	size := 8
	orig := makeData(nElems * size)
	in := codec.ArrayChunk{
		Shape: []int64{int64(nElems)},
		Dtype: dt,
		Data:  append([]byte(nil), orig...),
	}
	c := configure(t, "little")
	encoded, err := c.EncodeAB(in)
	if err != nil {
		t.Fatalf("EncodeAB: %v", err)
	}

	preallocated := make([]byte, nElems*size)
	origPtr := unsafe.SliceData(preallocated)

	out := codec.ArrayChunk{
		Shape: []int64{int64(nElems)},
		Dtype: dt,
		Data:  preallocated,
	}
	if err := c.DecodeAB(encoded, &out); err != nil {
		t.Fatalf("DecodeAB: %v", err)
	}
	if unsafe.SliceData(out.Data) != origPtr {
		t.Fatalf("DecodeAB reallocated despite sufficient cap")
	}
	if len(out.Data) != nElems*size {
		t.Fatalf("DecodeAB returned len %d, want %d", len(out.Data), nElems*size)
	}
	if !stdbytes.Equal(out.Data, orig) {
		t.Fatalf("DecodeAB into preallocated buffer produced wrong bytes")
	}

	// Allocates when nil.
	out2 := codec.ArrayChunk{Shape: []int64{int64(nElems)}, Dtype: dt}
	if err := c.DecodeAB(encoded, &out2); err != nil {
		t.Fatalf("DecodeAB(nil buf): %v", err)
	}
	if !stdbytes.Equal(out2.Data, orig) {
		t.Fatalf("DecodeAB(nil buf): wrong bytes")
	}
}

func TestDecodeAllocatesWhenInsufficient(t *testing.T) {
	const nElems = 8
	size := 4
	orig := makeData(nElems * size)
	in := codec.ArrayChunk{
		Shape: []int64{int64(nElems)},
		Dtype: "int32",
		Data:  append([]byte(nil), orig...),
	}
	c := configure(t, "little")
	encoded, err := c.EncodeAB(in)
	if err != nil {
		t.Fatalf("EncodeAB: %v", err)
	}
	tooSmall := make([]byte, 4) // way smaller than needed
	out := codec.ArrayChunk{
		Shape: []int64{int64(nElems)},
		Dtype: "int32",
		Data:  tooSmall,
	}
	if err := c.DecodeAB(encoded, &out); err != nil {
		t.Fatalf("DecodeAB: %v", err)
	}
	if len(out.Data) != nElems*size {
		t.Fatalf("len = %d, want %d", len(out.Data), nElems*size)
	}
	if !stdbytes.Equal(out.Data, orig) {
		t.Fatalf("decoded bytes differ from original")
	}
}

func TestDecodeSizeMismatch(t *testing.T) {
	c := configure(t, "little")
	out := codec.ArrayChunk{Shape: []int64{4}, Dtype: "int32"} // expects 16 bytes
	err := c.DecodeAB(make([]byte, 12), &out)
	if err == nil {
		t.Fatal("expected size-mismatch error, got nil")
	}
}

func TestUnsupportedDtype(t *testing.T) {
	c := configure(t, "little")
	_, err := c.EncodeAB(codec.ArrayChunk{Shape: []int64{1}, Dtype: "string", Data: []byte{0}})
	if err == nil {
		t.Fatal("EncodeAB: expected error for unknown dtype")
	}
	out := codec.ArrayChunk{Shape: []int64{1}, Dtype: "string"}
	if err := c.DecodeAB([]byte{0}, &out); err == nil {
		t.Fatal("DecodeAB: expected error for unknown dtype")
	}
}

func TestRegistered(t *testing.T) {
	f := codec.Lookup("bytes")
	if f == nil {
		t.Fatal("bytes codec not registered")
	}
	c := f()
	if _, ok := c.(codec.ArrayToBytesCodec); !ok {
		t.Fatalf("registered factory did not return ArrayToBytesCodec, got %T", c)
	}
	if c.Name() != "bytes" {
		t.Fatalf("Name() = %q, want \"bytes\"", c.Name())
	}
}
