package zarr

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"unsafe"
)

// parseFillValue decodes the JSON fill_value field into typed value bytes
// matching dt. The output is always elementSize bytes in native (little)
// encoding suitable for memcpy into a result buffer.
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
		switch size {
		case 1:
			out[0] = byte(int8(n))
		case 2:
			binary.LittleEndian.PutUint16(out, uint16(int16(n)))
		case 4:
			binary.LittleEndian.PutUint32(out, uint32(int32(n)))
		case 8:
			binary.LittleEndian.PutUint64(out, uint64(n))
		}
	case DtypeUint8, DtypeUint16, DtypeUint32, DtypeUint64:
		var n uint64
		if err := json.Unmarshal(raw, &n); err != nil {
			return nil, err
		}
		switch size {
		case 1:
			out[0] = byte(n)
		case 2:
			binary.LittleEndian.PutUint16(out, uint16(n))
		case 4:
			binary.LittleEndian.PutUint32(out, uint32(n))
		case 8:
			binary.LittleEndian.PutUint64(out, n)
		}
	case DtypeFloat16:
		f, err := parseSpecialFloat(raw)
		if err != nil {
			return nil, err
		}
		// Encode as float16 bits (simplified: use the float32→float16 path).
		bits := float32ToFloat16Bits(float32(f))
		binary.LittleEndian.PutUint16(out, bits)
	case DtypeFloat32:
		f, err := parseSpecialFloat(raw)
		if err != nil {
			return nil, err
		}
		binary.LittleEndian.PutUint32(out, math.Float32bits(float32(f)))
	case DtypeFloat64:
		f, err := parseSpecialFloat(raw)
		if err != nil {
			return nil, err
		}
		binary.LittleEndian.PutUint64(out, math.Float64bits(f))
	case DtypeComplex64, DtypeComplex128:
		var arr [2]json.RawMessage
		if err := json.Unmarshal(raw, &arr); err != nil {
			return nil, err
		}
		re, err := parseSpecialFloat(arr[0])
		if err != nil {
			return nil, err
		}
		im, err := parseSpecialFloat(arr[1])
		if err != nil {
			return nil, err
		}
		if dt == DtypeComplex64 {
			binary.LittleEndian.PutUint32(out[0:4], math.Float32bits(float32(re)))
			binary.LittleEndian.PutUint32(out[4:8], math.Float32bits(float32(im)))
		} else {
			binary.LittleEndian.PutUint64(out[0:8], math.Float64bits(re))
			binary.LittleEndian.PutUint64(out[8:16], math.Float64bits(im))
		}
	default:
		return nil, fmt.Errorf("fill value not supported for dtype %s", dt)
	}
	return out, nil
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

// float32ToFloat16Bits is a minimal IEEE-754 binary32 → binary16 conversion.
// Subnormals and overflow are clamped/flushed.
func float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int32((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff
	if exp <= 0 {
		// Underflow to zero (subnormals not handled exactly; M1 fill values
		// are typically zero/NaN/Inf).
		return sign
	}
	if exp >= 0x1f {
		if mant != 0 && (bits>>23)&0xff == 0xff {
			return sign | 0x7e00 // NaN
		}
		return sign | 0x7c00 // Inf
	}
	return sign | uint16(exp<<10) | uint16(mant>>13)
}

// fillValueToTyped reinterprets the little-endian fill bytes as one element of
// type T. Caller must ensure len(b) == sizeof(T).
func fillValueToTyped[T Numeric](b []byte) T {
	var t T
	_ = b[len(b)-1] // bounds check
	dst := unsafe.Slice((*byte)(unsafe.Pointer(&t)), int(unsafe.Sizeof(t)))
	copy(dst, b)
	return t
}
