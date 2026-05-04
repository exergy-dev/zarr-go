package bytes

import "testing"

const benchSize = 1 << 20 // 1 MiB

// scalarSwap32 mirrors the unrolled scalar branch in swapBytes for unit==4.
// Kept as a separate function so benchmarks can compare it directly against
// the SIMD path without the SIMD wiring inside swapBytes choosing the bulk
// helper.
func scalarSwap32(dst, src []byte) {
	for i := 0; i < len(src); i += 4 {
		dst[i] = src[i+3]
		dst[i+1] = src[i+2]
		dst[i+2] = src[i+1]
		dst[i+3] = src[i]
	}
}

func scalarSwap64(dst, src []byte) {
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

func BenchmarkSwap32_1MiB_Scalar(b *testing.B) {
	src := make([]byte, benchSize)
	dst := make([]byte, benchSize)
	for i := range src {
		src[i] = byte(i)
	}
	b.SetBytes(int64(benchSize))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scalarSwap32(dst, src)
	}
}

func BenchmarkSwap32_1MiB_SIMD(b *testing.B) {
	if swapBulk32(make([]byte, 32), make([]byte, 32)) == 0 {
		b.Skip("no SIMD bulk swap available on this build/CPU")
	}
	src := make([]byte, benchSize)
	dst := make([]byte, benchSize)
	for i := range src {
		src[i] = byte(i)
	}
	b.SetBytes(int64(benchSize))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		swapBytes(dst, src, 4)
	}
}

func BenchmarkSwap64_1MiB_Scalar(b *testing.B) {
	src := make([]byte, benchSize)
	dst := make([]byte, benchSize)
	for i := range src {
		src[i] = byte(i)
	}
	b.SetBytes(int64(benchSize))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scalarSwap64(dst, src)
	}
}

func BenchmarkSwap64_1MiB_SIMD(b *testing.B) {
	if swapBulk64(make([]byte, 32), make([]byte, 32)) == 0 {
		b.Skip("no SIMD bulk swap available on this build/CPU")
	}
	src := make([]byte, benchSize)
	dst := make([]byte, benchSize)
	for i := range src {
		src[i] = byte(i)
	}
	b.SetBytes(int64(benchSize))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		swapBytes(dst, src, 8)
	}
}
