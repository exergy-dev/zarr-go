// No SIMD bulk swap outside amd64: these return 0 so swapBytes runs the
// portable word loops (swapWords32/64) over the whole buffer.

//go:build !amd64

package bytes

func swapBulk32(dst, src []byte) int { return 0 }
func swapBulk64(dst, src []byte) int { return 0 }
