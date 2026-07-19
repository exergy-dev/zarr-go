package codec

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
