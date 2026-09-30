//go:build amd64

package bytes

import (
	stdbytes "bytes"
	"testing"

	"golang.org/x/sys/cpu"
)

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
		refSwap(want, src, 4)

		got := make([]byte, n)
		if n >= 32 {
			bulk := n &^ 31
			swap32AVX2(&got[0], &src[0], bulk)
			// Tail via scalar to fill the comparison.
			if bulk < n {
				refSwap(got[bulk:], src[bulk:], 4)
			}
		} else {
			refSwap(got, src, 4)
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
		refSwap(want, src, 8)

		got := make([]byte, n)
		if n >= 32 {
			bulk := n &^ 31
			swap64AVX2(&got[0], &src[0], bulk)
			if bulk < n {
				refSwap(got[bulk:], src[bulk:], 8)
			}
		} else {
			refSwap(got, src, 8)
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
		refSwap(want, src, 4)

		got := make([]byte, n)
		if n >= 16 {
			bulk := n &^ 15
			swap32SSSE3(&got[0], &src[0], bulk)
			if bulk < n {
				refSwap(got[bulk:], src[bulk:], 4)
			}
		} else {
			refSwap(got, src, 4)
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
		refSwap(want, src, 8)

		got := make([]byte, n)
		if n >= 16 {
			bulk := n &^ 15
			swap64SSSE3(&got[0], &src[0], bulk)
			if bulk < n {
				refSwap(got[bulk:], src[bulk:], 8)
			}
		} else {
			refSwap(got, src, 8)
		}
		if !stdbytes.Equal(got, want) {
			t.Fatalf("swap64SSSE3 n=%d mismatch", n)
		}
	}
}

// TestSwapBytesDispatchPaths runs the portable swapBytes check with the CPU
// feature flags forced to each dispatch path swapBulk32/64 can take: AVX2,
// SSSE3 only, and neither (pure-Go fallback). Paths the CPU cannot execute
// are skipped.
func TestSwapBytesDispatchPaths(t *testing.T) {
	origAVX2, origSSSE3 := hasAVX2, hasSSSE3
	t.Cleanup(func() { hasAVX2, hasSSSE3 = origAVX2, origSSSE3 })
	for _, tc := range []struct {
		name        string
		avx2, ssse3 bool
	}{
		{"avx2", true, true},
		{"ssse3", false, true},
		{"fallback", false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if (tc.avx2 && !origAVX2) || (tc.ssse3 && !origSSSE3) {
				t.Skip("not supported on this CPU")
			}
			hasAVX2, hasSSSE3 = tc.avx2, tc.ssse3
			checkSwapBytes(t)
		})
	}
}
