package codec

import "encoding/json"

// DtypeSize returns the byte size of one element of the named Zarr v3
// canonical dtype, or 0 if name is not recognized.
func DtypeSize(name string) int {
	switch name {
	case "bool", "int8", "uint8":
		return 1
	case "int16", "uint16", "float16":
		return 2
	case "int32", "uint32", "float32":
		return 4
	case "int64", "uint64", "float64", "complex64":
		return 8
	case "complex128":
		return 16
	}
	return 0
}

// DefaultFillValueJSON returns the spec default fill_value JSON for the
// named Zarr v3 canonical dtype when none was supplied: false for bool, a
// 2-element zero array for complex dtypes (which need one real+imaginary
// component each), 0 otherwise.
func DefaultFillValueJSON(name string) json.RawMessage {
	switch name {
	case "bool":
		return json.RawMessage("false")
	case "complex64", "complex128":
		return json.RawMessage("[0,0]")
	default:
		return json.RawMessage("0")
	}
}
