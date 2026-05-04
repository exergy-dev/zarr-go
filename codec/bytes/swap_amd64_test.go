//go:build amd64

package bytes

import (
	stdbytes "bytes"
	"testing"

	"golang.org/x/sys/cpu"
)

// scalarSwap is a reference implementation independent of swapBytes; used to
// validate the SIMD output.
func scalarSwap(dst, src []byte, unit int) {
	for i := 0; i < len(src); i += unit {
		for j := 0; j < unit; j++ {
			dst[i+j] = src[i+unit-1-j]
		}
	}
}

func makePattern(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*131 + 17)
	}
	return b
}

var swapSizes = []int{0, 8, 16, 32, 64, 96, 128, 1024, 4096, 1 << 20}

func TestSwap32AVX2MatchesScalar(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 not supported on this CPU")
	}
	for _, n := range swapSizes {
		// Round to multiple of 4.
		n &^= 3
		src := makePattern(n)
		want := make([]byte, n)
		scalarSwap(want, src, 4)

		got := make([]byte, n)
		if n >= 32 {
			bulk := n &^ 31
			swap32AVX2(&got[0], &src[0], bulk)
			// Tail via scalar to fill the comparison.
			if bulk < n {
				scalarSwap(got[bulk:], src[bulk:], 4)
			}
		} else {
			scalarSwap(got, src, 4)
		}
		if !stdbytes.Equal(got, want) {
			t.Fatalf("swap32AVX2 n=%d mismatch\n got=%x\nwant=%x", n, got, want)
		}
	}
}

func TestSwap64AVX2MatchesScalar(t *testing.T) {
	if !cpu.X86.HasAVX2 {
		t.Skip("AVX2 not supported on this CPU")
	}
	for _, n := range swapSizes {
		n &^= 7
		src := makePattern(n)
		want := make([]byte, n)
		scalarSwap(want, src, 8)

		got := make([]byte, n)
		if n >= 32 {
			bulk := n &^ 31
			swap64AVX2(&got[0], &src[0], bulk)
			if bulk < n {
				scalarSwap(got[bulk:], src[bulk:], 8)
			}
		} else {
			scalarSwap(got, src, 8)
		}
		if !stdbytes.Equal(got, want) {
			t.Fatalf("swap64AVX2 n=%d mismatch\n got=%x\nwant=%x", n, got, want)
		}
	}
}

func TestSwap32SSSE3MatchesScalar(t *testing.T) {
	if !cpu.X86.HasSSSE3 {
		t.Skip("SSSE3 not supported on this CPU")
	}
	for _, n := range swapSizes {
		n &^= 3
		src := makePattern(n)
		want := make([]byte, n)
		scalarSwap(want, src, 4)

		got := make([]byte, n)
		if n >= 16 {
			bulk := n &^ 15
			swap32SSSE3(&got[0], &src[0], bulk)
			if bulk < n {
				scalarSwap(got[bulk:], src[bulk:], 4)
			}
		} else {
			scalarSwap(got, src, 4)
		}
		if !stdbytes.Equal(got, want) {
			t.Fatalf("swap32SSSE3 n=%d mismatch", n)
		}
	}
}

func TestSwap64SSSE3MatchesScalar(t *testing.T) {
	if !cpu.X86.HasSSSE3 {
		t.Skip("SSSE3 not supported on this CPU")
	}
	for _, n := range swapSizes {
		n &^= 7
		src := makePattern(n)
		want := make([]byte, n)
		scalarSwap(want, src, 8)

		got := make([]byte, n)
		if n >= 16 {
			bulk := n &^ 15
			swap64SSSE3(&got[0], &src[0], bulk)
			if bulk < n {
				scalarSwap(got[bulk:], src[bulk:], 8)
			}
		} else {
			scalarSwap(got, src, 8)
		}
		if !stdbytes.Equal(got, want) {
			t.Fatalf("swap64SSSE3 n=%d mismatch", n)
		}
	}
}

// TestSwapBytesParity exercises the full swapBytes path (which selects the
// best available SIMD path internally) against the reference scalarSwap.
func TestSwapBytesParity(t *testing.T) {
	for _, unit := range []int{2, 4, 8, 16} {
		for _, n := range swapSizes {
			n -= n % unit
			src := makePattern(n)
			want := make([]byte, n)
			scalarSwap(want, src, unit)
			got := make([]byte, n)
			swapBytes(got, src, unit)
			if !stdbytes.Equal(got, want) {
				t.Fatalf("swapBytes unit=%d n=%d mismatch", unit, n)
			}
		}
	}
}
