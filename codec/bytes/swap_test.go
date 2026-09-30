package bytes

import (
	stdbytes "bytes"
	"testing"
)

// refSwap is a trivially-correct reference for swapBytes, independent of its
// fast paths: byte j of each unit-sized group comes from byte unit-1-j.
func refSwap(dst, src []byte, unit int) {
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

// maxSwapTestLen covers every length up to several multiples of the widest
// bulk step (64 bytes), so each bulk/tail split is exercised.
const maxSwapTestLen = 520

// checkSwapBytes compares swapBytes against refSwap for every swap unit the
// bytes codec uses (including the complex dtypes, which swap each component
// separately) and every valid length from 0 to maxSwapTestLen. It checks the
// output, that nothing past len(dst) is written, that src is untouched, and
// the documented in-place (dst == src) mode.
func checkSwapBytes(t *testing.T) {
	t.Helper()
	units := []struct {
		name string
		unit int
	}{
		{"int16", swapUnit("int16", 2)},
		{"float32", swapUnit("float32", 4)},
		{"float64", swapUnit("float64", 8)},
		{"complex64", swapUnit("complex64", 8)},
		{"complex128", swapUnit("complex128", 16)},
		{"generic16", 16},
	}
	const guard = 0xA5
	for _, u := range units {
		for n := 0; n <= maxSwapTestLen; n += u.unit {
			src := makePattern(n)
			want := make([]byte, n)
			refSwap(want, src, u.unit)

			buf := stdbytes.Repeat([]byte{guard}, n+u.unit)
			got := buf[:n]
			swapBytes(got, src, u.unit)
			if !stdbytes.Equal(got, want) {
				t.Fatalf("%s (unit %d) n=%d: mismatch\n got=%x\nwant=%x", u.name, u.unit, n, got, want)
			}
			for i, b := range buf[n:] {
				if b != guard {
					t.Fatalf("%s (unit %d) n=%d: wrote past len(dst) at +%d", u.name, u.unit, n, i)
				}
			}

			inPlace := stdbytes.Clone(src)
			swapBytes(inPlace, inPlace, u.unit)
			if !stdbytes.Equal(inPlace, want) {
				t.Fatalf("%s (unit %d) n=%d: in-place mismatch", u.name, u.unit, n)
			}
			if !stdbytes.Equal(src, makePattern(n)) {
				t.Fatalf("%s (unit %d) n=%d: src modified", u.name, u.unit, n)
			}
		}
	}
}

func TestSwapBytesMatchesReference(t *testing.T) {
	checkSwapBytes(t)
}

func TestSwapBytesEmpty(t *testing.T) {
	for _, unit := range []int{2, 4, 8, 16} {
		swapBytes(nil, nil, unit)
		swapBytes([]byte{}, []byte{}, unit)
	}
}

// TestSwapBytesLarge covers buffers large enough for the unrolled bulk loops
// to run many iterations, at a length that is not a multiple of any bulk
// width.
func TestSwapBytesLarge(t *testing.T) {
	const n = 1<<20 + 8*3
	src := makePattern(n)
	for _, unit := range []int{2, 4, 8} {
		want := make([]byte, n)
		refSwap(want, src, unit)
		got := make([]byte, n)
		swapBytes(got, src, unit)
		if !stdbytes.Equal(got, want) {
			t.Fatalf("unit %d n=%d: mismatch", unit, n)
		}
	}
}
