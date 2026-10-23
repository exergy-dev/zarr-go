package zarr

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// encodeFillValue renders a WithFillValue argument in its v3 metadata form:
// numbers as JSON numbers, non-finite floats as the strings "NaN",
// "Infinity" and "-Infinity", complex values as a two-element [real, imag]
// array, and a json.RawMessage unchanged.
func encodeFillValue(v any) (json.RawMessage, error) {
	switch x := v.(type) {
	case json.RawMessage:
		if len(x) == 0 {
			return nil, fmt.Errorf("fill value: empty json.RawMessage")
		}
		return append(json.RawMessage(nil), x...), nil
	case bool:
		return json.RawMessage(strconv.FormatBool(x)), nil
	case int:
		return json.RawMessage(strconv.FormatInt(int64(x), 10)), nil
	case int8:
		return json.RawMessage(strconv.FormatInt(int64(x), 10)), nil
	case int16:
		return json.RawMessage(strconv.FormatInt(int64(x), 10)), nil
	case int32:
		return json.RawMessage(strconv.FormatInt(int64(x), 10)), nil
	case int64:
		return json.RawMessage(strconv.FormatInt(x, 10)), nil
	case uint:
		return json.RawMessage(strconv.FormatUint(uint64(x), 10)), nil
	case uint8:
		return json.RawMessage(strconv.FormatUint(uint64(x), 10)), nil
	case uint16:
		return json.RawMessage(strconv.FormatUint(uint64(x), 10)), nil
	case uint32:
		return json.RawMessage(strconv.FormatUint(uint64(x), 10)), nil
	case uint64:
		return json.RawMessage(strconv.FormatUint(x, 10)), nil
	case Float16:
		return json.RawMessage(formatFloat(float64(Float16ToFloat32(x)), 32)), nil
	case float32:
		return json.RawMessage(formatFloat(float64(x), 32)), nil
	case float64:
		return json.RawMessage(formatFloat(x, 64)), nil
	case complex64:
		return json.RawMessage("[" + formatFloat(float64(real(x)), 32) + "," + formatFloat(float64(imag(x)), 32) + "]"), nil
	case complex128:
		return json.RawMessage("[" + formatFloat(real(x), 64) + "," + formatFloat(imag(x), 64) + "]"), nil
	}
	return nil, fmt.Errorf("fill value: unsupported type %T", v)
}

// formatFloat renders f as a v3 fill value: a JSON number, or the quoted
// string the spec uses for a non-finite value.
func formatFloat(f float64, bitSize int) string {
	switch {
	case math.IsNaN(f):
		return `"NaN"`
	case math.IsInf(f, 1):
		return `"Infinity"`
	case math.IsInf(f, -1):
		return `"-Infinity"`
	}
	return strconv.FormatFloat(f, 'g', -1, bitSize)
}

// parseFillValue decodes the JSON fill_value field into typed value bytes
// matching dt. The output is always elementSize bytes in native byte order,
// suitable for memcpy into a result buffer.
func parseFillValue(raw json.RawMessage, dt Dtype) ([]byte, error) {
	size := dt.Size()
	out := make([]byte, size)
	if len(raw) == 0 || string(raw) == "null" {
		return out, nil
	}
	switch dt {
	case DtypeBool:
		var b bool
		if err := json.Unmarshal(raw, &b); err != nil {
			return nil, err
		}
		if b {
			out[0] = 1
		}
	case DtypeInt8, DtypeInt16, DtypeInt32, DtypeInt64:
		var n int64
		if err := json.Unmarshal(raw, &n); err != nil {
			return nil, err
		}
		if bits := uint(size * 8); bits < 64 && (n < -1<<(bits-1) || n >= 1<<(bits-1)) {
			return nil, fmt.Errorf("fill value %d out of range for dtype %s", n, dt)
		}
		switch size {
		case 1:
			out[0] = byte(int8(n))
		case 2:
			binary.NativeEndian.PutUint16(out, uint16(int16(n)))
		case 4:
			binary.NativeEndian.PutUint32(out, uint32(int32(n)))
		case 8:
			binary.NativeEndian.PutUint64(out, uint64(n))
		}
	case DtypeUint8, DtypeUint16, DtypeUint32, DtypeUint64:
		var n uint64
		if err := json.Unmarshal(raw, &n); err != nil {
			if len(raw) > 0 && raw[0] == '-' {
				return nil, fmt.Errorf("fill value %s out of range for dtype %s", raw, dt)
			}
			return nil, err
		}
		if bits := uint(size * 8); bits < 64 && n>>bits != 0 {
			return nil, fmt.Errorf("fill value %d out of range for dtype %s", n, dt)
		}
		switch size {
		case 1:
			out[0] = byte(n)
		case 2:
			binary.NativeEndian.PutUint16(out, uint16(n))
		case 4:
			binary.NativeEndian.PutUint32(out, uint32(n))
		case 8:
			binary.NativeEndian.PutUint64(out, n)
		}
	case DtypeFloat16:
		bits, err := parseFloatBits(raw, 16)
		if err != nil {
			return nil, err
		}
		binary.NativeEndian.PutUint16(out, uint16(bits))
	case DtypeFloat32:
		bits, err := parseFloatBits(raw, 32)
		if err != nil {
			return nil, err
		}
		binary.NativeEndian.PutUint32(out, uint32(bits))
	case DtypeFloat64:
		bits, err := parseFloatBits(raw, 64)
		if err != nil {
			return nil, err
		}
		binary.NativeEndian.PutUint64(out, bits)
	case DtypeComplex64, DtypeComplex128:
		var arr [2]json.RawMessage
		if err := json.Unmarshal(raw, &arr); err != nil {
			return nil, err
		}
		width := 32
		if dt == DtypeComplex128 {
			width = 64
		}
		reBits, err := parseFloatBits(arr[0], width)
		if err != nil {
			return nil, err
		}
		imBits, err := parseFloatBits(arr[1], width)
		if err != nil {
			return nil, err
		}
		if dt == DtypeComplex64 {
			binary.NativeEndian.PutUint32(out[0:4], uint32(reBits))
			binary.NativeEndian.PutUint32(out[4:8], uint32(imBits))
		} else {
			binary.NativeEndian.PutUint64(out[0:8], reBits)
			binary.NativeEndian.PutUint64(out[8:16], imBits)
		}
	default:
		return nil, fmt.Errorf("fill value not supported for dtype %s", dt)
	}
	return out, nil
}

// parseFloatBits parses one float fill-value component — a JSON number,
// a "NaN"/"Infinity"/"-Infinity" string, or a v3 hex bit-pattern string like
// "0x7fc00000" — into its raw bits at the given width (16, 32, or 64).
// Hex bit-pattern values are decoded directly (no float64 round-trip), so
// exact NaN payloads and signs are preserved.
func parseFloatBits(raw json.RawMessage, bitWidth int) (uint64, error) {
	if bits, ok, err := hexFloatBits(raw, bitWidth); ok {
		return bits, err
	}
	f, err := parseSpecialFloat(raw)
	if err != nil {
		return 0, err
	}
	switch bitWidth {
	case 16:
		return uint64(float32ToFloat16Bits(float32(f))), nil
	case 32:
		return uint64(math.Float32bits(float32(f))), nil
	default:
		return math.Float64bits(f), nil
	}
}

// hexFloatBits parses a v3 hex bit-pattern fill value string ("0x..." or
// "0X...") into its raw bits, sized to bitWidth (16, 32, or 64). ok is false
// when raw isn't a hex-string fill value, so the caller falls through to the
// numeric/NaN/Infinity path; err is only meaningful when ok is true.
func hexFloatBits(raw json.RawMessage, bitWidth int) (bits uint64, ok bool, err error) {
	if len(raw) < 3 || raw[0] != '"' {
		return 0, false, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return 0, false, nil
	}
	if !strings.HasPrefix(s, "0x") && !strings.HasPrefix(s, "0X") {
		return 0, false, nil
	}
	v, err := strconv.ParseUint(s[2:], 16, bitWidth)
	if err != nil {
		return 0, true, fmt.Errorf("invalid hex float fill value %q: %w", s, err)
	}
	return v, true, nil
}

// parseSpecialFloat handles "NaN", "Infinity", "-Infinity" strings as well as
// numbers per the v3 spec.
func parseSpecialFloat(raw json.RawMessage) (float64, error) {
	if len(raw) > 0 && raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return 0, err
		}
		switch s {
		case "NaN":
			return math.NaN(), nil
		case "Infinity":
			return math.Inf(1), nil
		case "-Infinity":
			return math.Inf(-1), nil
		}
		return 0, fmt.Errorf("unknown special float %q", s)
	}
	var f float64
	if err := json.Unmarshal(raw, &f); err != nil {
		return 0, err
	}
	return f, nil
}

// float32ToFloat16Bits is an IEEE-754 binary32 → binary16 conversion with
// round-to-nearest-even, including subnormal results. Values too large for
// binary16 become ±Inf; NaN stays NaN (quiet) and the sign is preserved.
func float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int(bits>>23) & 0xff
	mant := bits & 0x7fffff
	switch e := exp - 127 + 15; {
	case exp == 0xff && mant != 0:
		return sign | 0x7e00 // NaN
	case e >= 0x1f:
		return sign | 0x7c00 // Inf or overflow
	case e > 0:
		// Normal: rounding may carry into the exponent, up to Inf.
		return sign | uint16(roundShiftRNE(uint32(e)<<23|mant, 13))
	case e >= -10:
		// Subnormal: shift the explicit leading bit into the mantissa.
		// Rounding up to 0x0400 correctly yields the smallest normal.
		return sign | uint16(roundShiftRNE(mant|0x800000, uint(14-e)))
	default:
		return sign // below half the smallest subnormal: ±0
	}
}

// roundShiftRNE returns v >> shift rounded to nearest, ties to even.
func roundShiftRNE(v uint32, shift uint) uint32 {
	r, rem, half := v>>shift, v&(1<<shift-1), uint32(1)<<(shift-1)
	if rem > half || (rem == half && r&1 == 1) {
		r++
	}
	return r
}
