// Stub fast-path helpers for non-amd64 builds. Always return 0 so swapBytes
// uses the scalar implementation in full.

//go:build !amd64

package bytes

func swapBulk32(dst, src []byte) int { return 0 }
func swapBulk64(dst, src []byte) int { return 0 }
