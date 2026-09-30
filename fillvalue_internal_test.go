package zarr

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"slices"
	"testing"
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
