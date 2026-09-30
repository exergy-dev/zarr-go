package zarr

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"slices"
	"testing"
	"unsafe"
)

// nativeBytesFromLE converts a little-endian element encoding to the host's
// native byte order (fill bytes are memcpy'd into T, so they're native).
func nativeBytesFromLE(le []byte) []byte {
	out := slices.Clone(le)
	if binary.NativeEndian.Uint16([]byte{1, 0}) != 1 {
		slices.Reverse(out)
	}
	return out
}

// TestParseFillValueIntRange checks integer fill values are range-checked
// against the dtype's width: min/max are accepted with exact bytes, and
// anything outside the range is rejected instead of silently truncated.
func TestParseFillValueIntRange(t *testing.T) {
	cases := []struct {
		dt   Dtype
		raw  string
		want []byte // nil means an error is expected
	}{
		{DtypeInt8, `-128`, []byte{0x80}},
		{DtypeInt8, `127`, []byte{0x7f}},
		{DtypeInt8, `-129`, nil},
		{DtypeInt8, `128`, nil},
		{DtypeInt8, `300`, nil},
		{DtypeInt16, `-32768`, []byte{0x00, 0x80}},
		{DtypeInt16, `32767`, []byte{0xff, 0x7f}},
		{DtypeInt16, `-32769`, nil},
		{DtypeInt16, `32768`, nil},
		{DtypeInt32, `-2147483648`, []byte{0x00, 0x00, 0x00, 0x80}},
		{DtypeInt32, `2147483647`, []byte{0xff, 0xff, 0xff, 0x7f}},
		{DtypeInt32, `-2147483649`, nil},
		{DtypeInt32, `2147483648`, nil},
		{DtypeInt64, `-9223372036854775808`, []byte{0, 0, 0, 0, 0, 0, 0, 0x80}},
		{DtypeInt64, `9223372036854775807`, []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f}},
		{DtypeInt64, `-9223372036854775809`, nil},
		{DtypeInt64, `9223372036854775808`, nil},
		{DtypeUint8, `0`, []byte{0x00}},
		{DtypeUint8, `255`, []byte{0xff}},
		{DtypeUint8, `256`, nil},
		{DtypeUint8, `-1`, nil},
		{DtypeUint16, `65535`, []byte{0xff, 0xff}},
		{DtypeUint16, `65536`, nil},
		{DtypeUint16, `70000`, nil},
		{DtypeUint32, `4294967295`, []byte{0xff, 0xff, 0xff, 0xff}},
		{DtypeUint32, `4294967296`, nil},
		{DtypeUint64, `18446744073709551615`, []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}},
		{DtypeUint64, `18446744073709551616`, nil},
		{DtypeUint64, `-1`, nil},
	}
	for _, tc := range cases {
		t.Run(tc.dt.String()+"/"+tc.raw, func(t *testing.T) {
			got, err := parseFillValue(json.RawMessage(tc.raw), tc.dt)
			if tc.want == nil {
				if err == nil {
					t.Fatalf("parseFillValue(%s, %s) = %x, want error", tc.raw, tc.dt, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseFillValue(%s, %s): %v", tc.raw, tc.dt, err)
			}
			if !bytes.Equal(got, nativeBytesFromLE(tc.want)) {
				t.Fatalf("parseFillValue(%s, %s) = %x, want %x", tc.raw, tc.dt, got, tc.want)
			}
		})
	}
}

// memBytes returns the in-memory (host byte order) representation of v.
func memBytes[T any](v T) []byte {
	return slices.Clone(unsafe.Slice((*byte)(unsafe.Pointer(&v)), unsafe.Sizeof(v)))
}

// TestParseFillValueNativeByteOrder checks the fill bytes are the host's
// in-memory representation of the typed value, since fillTyped memcpys them
// straight into T. (Only distinguishable from little-endian on big-endian
// hosts.)
func TestParseFillValueNativeByteOrder(t *testing.T) {
	cases := []struct {
		dt   Dtype
		raw  string
		want []byte
	}{
		{DtypeInt16, `-2`, memBytes(int16(-2))},
		{DtypeInt32, `305419896`, memBytes(int32(0x12345678))},
		{DtypeInt64, `-81985529216486896`, memBytes(int64(-0x123456789abcdf0))},
		{DtypeUint16, `4660`, memBytes(uint16(0x1234))},
		{DtypeUint32, `305419896`, memBytes(uint32(0x12345678))},
		{DtypeUint64, `81985529216486895`, memBytes(uint64(0x123456789abcdef))},
		{DtypeFloat16, `1.5`, memBytes(Float16(0x3e00))},
		{DtypeFloat32, `1.5`, memBytes(float32(1.5))},
		{DtypeFloat64, `1.5`, memBytes(float64(1.5))},
		{DtypeComplex64, `[1.5, -2]`, memBytes(complex64(complex(1.5, -2)))},
		{DtypeComplex128, `[1.5, -2]`, memBytes(complex(1.5, -2))},
	}
	for _, tc := range cases {
		t.Run(tc.dt.String(), func(t *testing.T) {
			got, err := parseFillValue(json.RawMessage(tc.raw), tc.dt)
			if err != nil {
				t.Fatalf("parseFillValue(%s, %s): %v", tc.raw, tc.dt, err)
			}
			if !bytes.Equal(got, tc.want) {
				t.Fatalf("parseFillValue(%s, %s) = %x, want native %x", tc.raw, tc.dt, got, tc.want)
			}
		})
	}
}

// TestFloat32ToFloat16Bits checks binary32 → binary16 conversion rounds to
// nearest-even (including into and out of the subnormal range) and preserves
// specials. Expected bits match numpy's float16.
func TestFloat32ToFloat16Bits(t *testing.T) {
	cases := []struct {
		in   float32
		want uint16
	}{
		{0, 0x0000},
		{float32(math.Copysign(0, -1)), 0x8000},
		{1, 0x3C00},
		{-2, 0xC000},
		{0.1, 0x2E66},
		{0.3, 0x34CD},                     // remainder above half: rounds up
		{1 + 1.0/2048, 0x3C00},            // exact tie, even stays
		{1 + 3.0/2048, 0x3C02},            // exact tie, odd rounds up to even
		{1 + 1.0/2048 + 1.0/4096, 0x3C01}, // above tie
		{65504, 0x7BFF},                   // max finite
		{65519, 0x7BFF},                   // just below the overflow tie
		{65520, 0x7C00},                   // tie rounds up to +Inf
		{-65520, 0xFC00},
		{1e6, 0x7C00},
		{6.103515625e-05, 0x0400}, // 2^-14, min normal
		{6.1e-05, 0x03FF},         // largest subnormal region
		{6.1032e-05, 0x0400},      // subnormal rounding up into normal
		{6e-8, 0x0001},            // ~min subnormal 2^-24
		{-6e-8, 0x8001},
		{2.9802322e-08, 0x0000}, // 2^-25 exact tie, rounds to even 0
		{3e-8, 0x0001},          // just above 2^-25
		{1e-10, 0x0000},
		{float32(math.Inf(1)), 0x7C00},
		{float32(math.Inf(-1)), 0xFC00},
	}
	for _, tc := range cases {
		if got := float32ToFloat16Bits(tc.in); got != tc.want {
			t.Errorf("float32ToFloat16Bits(%g) = %#04x, want %#04x", tc.in, got, tc.want)
		}
	}
	if got := float32ToFloat16Bits(float32(math.NaN())); got&0x7C00 != 0x7C00 || got&0x03FF == 0 {
		t.Errorf("float32ToFloat16Bits(NaN) = %#04x, want a NaN", got)
	}
	if got := float32ToFloat16Bits(-float32(math.NaN())); got&0x8000 == 0 {
		t.Errorf("float32ToFloat16Bits(-NaN) = %#04x, want sign preserved", got)
	}
}
