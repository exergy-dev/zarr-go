package zarr

import (
	"fmt"
	"math"

	"github.com/exergy-dev/zarr-go/codec"
)

// Float16 is the IEEE 754 binary16 type. Stored as the raw bit pattern;
// conversion helpers (Float16ToFloat32, float32ToFloat16Bits) live in this
// file and fillvalue.go.
type Float16 uint16

// Complex64 and Complex128 are aliases for the standard library complex types.
type (
	Complex64  = complex64
	Complex128 = complex128
)

// Numeric is the type constraint for all dtypes the typed Array[T] supports.
//
// bool is included to match the Zarr v3 spec; one byte per element.
//
// Deliberately exact (no ~), not approximate: DtypeOf and the byte-level
// codecs (bytes, transpose, ...) all switch/reinterpret on T's concrete
// type, not its underlying type, so a defined type like `type MyF32
// float32` would satisfy an approximate constraint but silently misbehave
// (DtypeOf would return DtypeInvalid instead of DtypeFloat32). Excluding
// such types from the constraint turns that into a compile error instead.
type Numeric interface {
	bool |
		int8 | int16 | int32 | int64 |
		uint8 | uint16 | uint32 | uint64 |
		Float16 | float32 | float64 |
		complex64 | complex128
}

// Dtype identifies the on-wire element type of a Zarr array.
type Dtype uint8

const (
	DtypeInvalid Dtype = iota
	DtypeBool
	DtypeInt8
	DtypeInt16
	DtypeInt32
	DtypeInt64
	DtypeUint8
	DtypeUint16
	DtypeUint32
	DtypeUint64
	DtypeFloat16
	DtypeFloat32
	DtypeFloat64
	DtypeComplex64
	DtypeComplex128
)

// Endian identifies the byte order of multi-byte dtypes.
type Endian uint8

const (
	EndianNative Endian = iota
	EndianLittle
	EndianBig
)

// Size returns the byte size of one element of the dtype.
func (d Dtype) Size() int {
	return codec.DtypeSize(d.String())
}

// String returns the v3 spec name for the dtype.
func (d Dtype) String() string {
	switch d {
	case DtypeBool:
		return "bool"
	case DtypeInt8:
		return "int8"
	case DtypeInt16:
		return "int16"
	case DtypeInt32:
		return "int32"
	case DtypeInt64:
		return "int64"
	case DtypeUint8:
		return "uint8"
	case DtypeUint16:
		return "uint16"
	case DtypeUint32:
		return "uint32"
	case DtypeUint64:
		return "uint64"
	case DtypeFloat16:
		return "float16"
	case DtypeFloat32:
		return "float32"
	case DtypeFloat64:
		return "float64"
	case DtypeComplex64:
		return "complex64"
	case DtypeComplex128:
		return "complex128"
	}
	return "invalid"
}

// ParseDtype maps a v3 spec dtype name to a Dtype.
func ParseDtype(name string) (Dtype, error) {
	for d := DtypeBool; d <= DtypeComplex128; d++ {
		if d.String() == name {
			return d, nil
		}
	}
	return DtypeInvalid, fmt.Errorf("%w: %q", ErrUnsupportedDtype, name)
}

// DtypeOf returns the Dtype matching the Go type T.
func DtypeOf[T Numeric]() Dtype {
	var z T
	switch any(z).(type) {
	case bool:
		return DtypeBool
	case int8:
		return DtypeInt8
	case int16:
		return DtypeInt16
	case int32:
		return DtypeInt32
	case int64:
		return DtypeInt64
	case uint8:
		return DtypeUint8
	case uint16:
		return DtypeUint16
	case uint32:
		return DtypeUint32
	case uint64:
		return DtypeUint64
	case Float16:
		return DtypeFloat16
	case float32:
		return DtypeFloat32
	case float64:
		return DtypeFloat64
	case complex64:
		return DtypeComplex64
	case complex128:
		return DtypeComplex128
	}
	return DtypeInvalid
}

// Float16ToFloat32 expands a half-precision float to single precision.
func Float16ToFloat32(h Float16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		// subnormal
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
	case 0x1f:
		return math.Float32frombits((sign << 31) | 0x7f800000 | (mant << 13))
	}
	return math.Float32frombits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
}
