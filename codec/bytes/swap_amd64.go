// SIMD byte-swap fast paths for amd64.
//
// These routines reverse fixed-width groups of bytes (4 or 8) using PSHUFB
// (SSSE3) or VPSHUFB (AVX2). They are wired in from swapBytes via the
// swapBulk32 / swapBulk64 helpers below; on CPUs without the required
// extension they return 0 and swapBytes uses the portable swapWords32/64.

//go:build amd64

package bytes

import "golang.org/x/sys/cpu"

//go:noescape
func swap32AVX2(dst, src *byte, n int)

//go:noescape
func swap64AVX2(dst, src *byte, n int)

//go:noescape
func swap32SSSE3(dst, src *byte, n int)

//go:noescape
func swap64SSSE3(dst, src *byte, n int)

// hasAVX2 / hasSSSE3 are detected once at package init.
var (
	hasAVX2  = cpu.X86.HasAVX2
	hasSSSE3 = cpu.X86.HasSSSE3
)

// swapBulk32 swaps the largest 32-aligned (AVX2) or 16-aligned (SSSE3) prefix
// of src into dst in 4-byte groups, returning the number of bytes consumed.
// The caller handles the remainder with swapWords32.
func swapBulk32(dst, src []byte) int {
	n := len(src)
	if n < 4 {
		return 0
	}
	switch {
	case hasAVX2:
		bulk := n &^ 31
		if bulk >= 32 {
			swap32AVX2(&dst[0], &src[0], bulk)
			return bulk
		}
		// Less than 32 bytes available; fall through to SSSE3 path if any.
		fallthrough
	case hasSSSE3:
		bulk := n &^ 15
		if bulk >= 16 {
			swap32SSSE3(&dst[0], &src[0], bulk)
			return bulk
		}
	}
	return 0
}

// swapBulk64 is the 8-byte-group analogue of swapBulk32.
func swapBulk64(dst, src []byte) int {
	n := len(src)
	if n < 8 {
		return 0
	}
	switch {
	case hasAVX2:
		bulk := n &^ 31
		if bulk >= 32 {
			swap64AVX2(&dst[0], &src[0], bulk)
			return bulk
		}
		fallthrough
	case hasSSSE3:
		bulk := n &^ 15
		if bulk >= 16 {
			swap64SSSE3(&dst[0], &src[0], bulk)
			return bulk
		}
	}
	return 0
}
