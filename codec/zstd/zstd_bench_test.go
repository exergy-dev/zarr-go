package zstd

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand"
	"testing"
)

const benchSize = 1 << 20 // 1 MiB

// benchSizes are a small chunk and a typical (1 MiB) chunk.
var benchSizes = []struct {
	name string
	n    int
}{
	{"64KiB", 64 << 10},
	{"1MiB", benchSize},
}

// benchPayload returns n bytes of little-endian float32s: a slow ramp plus
// small quantized noise, which compresses roughly like real gridded data.
func benchPayload(n int) []byte {
	rng := rand.New(rand.NewSource(1))
	buf := make([]byte, n)
	for i := 0; i < n/4; i++ {
		v := float32(i>>8) + float32(rng.Intn(8))*0.125
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func newBenchCodec(b *testing.B) *Codec {
	b.Helper()
	c, err := New(json.RawMessage(`{"level":3}`))
	if err != nil {
		b.Fatal(err)
	}
	return c
}

// BenchmarkDecodeBBParallel measures DecodeBB throughput from many
// goroutines sharing one Codec, as Array.Read's decode workers do.
// Run with -cpu 1,4,8 to check that throughput scales.
func BenchmarkDecodeBBParallel(b *testing.B) {
	c := newBenchCodec(b)
	enc, err := c.EncodeBB(benchPayload(benchSize))
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(benchSize)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := c.DecodeBB(enc); err != nil {
				b.Error(err)
				return
			}
		}
	})
}

// BenchmarkEncodeBBParallel measures EncodeBB throughput from many
// goroutines sharing one Codec.
func BenchmarkEncodeBBParallel(b *testing.B) {
	c := newBenchCodec(b)
	in := benchPayload(benchSize)
	if _, err := c.EncodeBB(in); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(benchSize)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := c.EncodeBB(in); err != nil {
				b.Error(err)
				return
			}
		}
	})
}

func BenchmarkEncodeBB(b *testing.B) {
	for _, s := range benchSizes {
		b.Run(s.name, func(b *testing.B) {
			c := newBenchCodec(b)
			in := benchPayload(s.n)
			if _, err := c.EncodeBB(in); err != nil {
				b.Fatal(err)
			}
			b.SetBytes(int64(s.n))
			b.ReportAllocs()
			for b.Loop() {
				if _, err := c.EncodeBB(in); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkDecodeBB(b *testing.B) {
	for _, s := range benchSizes {
		b.Run(s.name, func(b *testing.B) {
			c := newBenchCodec(b)
			enc, err := c.EncodeBB(benchPayload(s.n))
			if err != nil {
				b.Fatal(err)
			}
			b.SetBytes(int64(s.n))
			b.ReportAllocs()
			for b.Loop() {
				if _, err := c.DecodeBB(enc); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
