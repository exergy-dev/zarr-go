package bytes

import (
	stdbytes "bytes"
	"encoding/binary"
	"fmt"
	"math/bits"
	"testing"
)

// This file benchmarks swapBytes (which dispatches the unit==4/8 bulk to the
// SIMD assembly on amd64) against the best portable Go alternatives, so the
// cost of keeping swap_amd64.s can be re-evaluated as the Go compiler improves.
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

// bytewiseSwap32 / bytewiseSwap64 are the byte-at-a-time loops swapBytes uses
// for the tail, and for the whole buffer on architectures without assembly.
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

// wordSwap32 is the fastest portable 4-byte swap found: 64-bit loads, BSWAP
// plus a 32-bit rotate (reversing both halves at once), unrolled 4x. Wider
// unrolling measured slower.
func wordSwap32(dst, src []byte) {
	n := len(src) &^ 31
	dst = dst[:len(src)]
	i := 0
	for ; i < n; i += 32 {
		s := src[i : i+32 : i+32]
		d := dst[i : i+32 : i+32]
		v0 := binary.LittleEndian.Uint64(s[0:])
		v1 := binary.LittleEndian.Uint64(s[8:])
		v2 := binary.LittleEndian.Uint64(s[16:])
		v3 := binary.LittleEndian.Uint64(s[24:])
		binary.LittleEndian.PutUint64(d[0:], bits.RotateLeft64(bits.ReverseBytes64(v0), 32))
		binary.LittleEndian.PutUint64(d[8:], bits.RotateLeft64(bits.ReverseBytes64(v1), 32))
		binary.LittleEndian.PutUint64(d[16:], bits.RotateLeft64(bits.ReverseBytes64(v2), 32))
		binary.LittleEndian.PutUint64(d[24:], bits.RotateLeft64(bits.ReverseBytes64(v3), 32))
	}
	for ; i+4 <= len(src); i += 4 {
		binary.LittleEndian.PutUint32(dst[i:], binary.BigEndian.Uint32(src[i:]))
	}
}

// wordSwap64 is the fastest portable 8-byte swap found: big-endian 64-bit
// loads (MOVQ+BSWAP), unrolled 8x.
func wordSwap64(dst, src []byte) {
	n := len(src) &^ 63
	dst = dst[:len(src)]
	i := 0
	for ; i < n; i += 64 {
		s := src[i : i+64 : i+64]
		d := dst[i : i+64 : i+64]
		v0 := binary.BigEndian.Uint64(s[0:])
		v1 := binary.BigEndian.Uint64(s[8:])
		v2 := binary.BigEndian.Uint64(s[16:])
		v3 := binary.BigEndian.Uint64(s[24:])
		v4 := binary.BigEndian.Uint64(s[32:])
		v5 := binary.BigEndian.Uint64(s[40:])
		v6 := binary.BigEndian.Uint64(s[48:])
		v7 := binary.BigEndian.Uint64(s[56:])
		binary.LittleEndian.PutUint64(d[0:], v0)
		binary.LittleEndian.PutUint64(d[8:], v1)
		binary.LittleEndian.PutUint64(d[16:], v2)
		binary.LittleEndian.PutUint64(d[24:], v3)
		binary.LittleEndian.PutUint64(d[32:], v4)
		binary.LittleEndian.PutUint64(d[40:], v5)
		binary.LittleEndian.PutUint64(d[48:], v6)
		binary.LittleEndian.PutUint64(d[56:], v7)
	}
	for ; i+8 <= len(src); i += 8 {
		binary.LittleEndian.PutUint64(dst[i:], binary.BigEndian.Uint64(src[i:]))
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
			{"words", wordSwap32},
			{"bytewise", bytewiseSwap32},
		}},
		{8, []impl{
			{"asm", func(d, s []byte) { swapBytes(d, s, 8) }},
			{"words", wordSwap64},
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
