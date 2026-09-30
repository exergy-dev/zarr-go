package bytes

import (
	stdbytes "bytes"
	"fmt"
	"testing"
)

// This file benchmarks swapBytes (impl=asm: the unit==4/8 bulk goes to the
// SIMD assembly on amd64) against the portable word loops it falls back to
// (impl=words: swapWords32/64, used on non-amd64 and on amd64 without SSSE3)
// and a plain byte loop (impl=bytewise), so the cost of keeping swap_amd64.s
// can be re-evaluated as the Go compiler improves.
//
// Compare with benchstat, e.g.
//
//	go test -run '^$' -bench BenchmarkSwap -count=8 ./codec/bytes > swap.txt
//	benchstat -col /impl swap.txt
//
// On an i9-14900KF (Go 1.25, GOAMD64=v1, 8 interleaved runs) the portable
// word loops reached this fraction of the assembly's throughput, which is
// below the 50%-at-64KiB bar for deleting it, so the assembly stays:
//
//	         4 KiB   64 KiB   1 MiB
//	unit=4    19%     40%      55%
//	unit=8    24%     48%      68%

// bytewiseSwap32 / bytewiseSwap64 are the byte-at-a-time loops swapBytes used
// for the tail and for non-amd64 before swapWords32/64, kept as a baseline.
func bytewiseSwap32(dst, src []byte) {
	for i := 0; i < len(src); i += 4 {
		dst[i] = src[i+3]
		dst[i+1] = src[i+2]
		dst[i+2] = src[i+1]
		dst[i+3] = src[i]
	}
}

func bytewiseSwap64(dst, src []byte) {
	for i := 0; i < len(src); i += 8 {
		dst[i] = src[i+7]
		dst[i+1] = src[i+6]
		dst[i+2] = src[i+5]
		dst[i+3] = src[i+4]
		dst[i+4] = src[i+3]
		dst[i+5] = src[i+2]
		dst[i+6] = src[i+1]
		dst[i+7] = src[i]
	}
}

func BenchmarkSwap(b *testing.B) {
	type impl struct {
		name string
		fn   func(dst, src []byte)
	}
	units := []struct {
		unit  int
		impls []impl
	}{
		{4, []impl{
			{"asm", func(d, s []byte) { swapBytes(d, s, 4) }},
			{"words", swapWords32},
			{"bytewise", bytewiseSwap32},
		}},
		{8, []impl{
			{"asm", func(d, s []byte) { swapBytes(d, s, 8) }},
			{"words", swapWords64},
			{"bytewise", bytewiseSwap64},
		}},
	}
	sizes := []struct {
		name string
		n    int
	}{{"4KiB", 4 << 10}, {"64KiB", 64 << 10}, {"1MiB", 1 << 20}}

	for _, u := range units {
		for _, sz := range sizes {
			src := make([]byte, sz.n)
			for i := range src {
				src[i] = byte(i*131 + 17)
			}
			want := make([]byte, sz.n)
			swapBytes(want, src, u.unit)
			for _, im := range u.impls {
				name := fmt.Sprintf("unit=%d/size=%s/impl=%s", u.unit, sz.name, im.name)
				b.Run(name, func(b *testing.B) {
					dst := make([]byte, sz.n)
					im.fn(dst, src)
					if !stdbytes.Equal(dst, want) {
						b.Fatalf("%s output differs from swapBytes", im.name)
					}
					b.SetBytes(int64(sz.n))
					for b.Loop() {
						im.fn(dst, src)
					}
				})
			}
		}
	}
}
