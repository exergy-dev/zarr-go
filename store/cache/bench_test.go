package cache_test

import (
	"context"
	"testing"

	"github.com/exergy-dev/zarr-go/store/cache"
	"github.com/exergy-dev/zarr-go/store/mem"
)

// BenchmarkCacheWarmHit measures the cost of a cached Get with a small
// payload, so the result reflects wrapper overhead (LRU bookkeeping +
// defensive copy) rather than memcpy throughput.
func BenchmarkCacheWarmHit(b *testing.B) {
	ctx := context.Background()
	inner := mem.New()
	payload := []byte("hello")
	if err := inner.Set(ctx, "k", payload); err != nil {
		b.Fatal(err)
	}
	c := cache.NewWriter(inner, cache.Config{})
	// Prime the cache.
	if _, err := c.Get(ctx, "k"); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := c.Get(ctx, "k")
		if err != nil {
			b.Fatal(err)
		}
		if len(v) != len(payload) {
			b.Fatalf("len %d", len(v))
		}
	}
}

// BenchmarkCacheWarmHit4K mirrors the above with a realistic chunk-sized
// payload so the copy cost is visible.
func BenchmarkCacheWarmHit4K(b *testing.B) {
	ctx := context.Background()
	inner := mem.New()
	payload := make([]byte, 4096)
	if err := inner.Set(ctx, "k", payload); err != nil {
		b.Fatal(err)
	}
	c := cache.NewWriter(inner, cache.Config{})
	if _, err := c.Get(ctx, "k"); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := c.Get(ctx, "k")
		if err != nil {
			b.Fatal(err)
		}
		if len(v) != 4096 {
			b.Fatalf("len %d", len(v))
		}
	}
}

// BenchmarkCacheMiss measures the cost of a Get that always misses (key
// absent in underlying): the flight and LRU bookkeeping overhead.
func BenchmarkCacheMiss(b *testing.B) {
	ctx := context.Background()
	inner := mem.New()
	c := cache.NewWriter(inner, cache.Config{})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = c.Get(ctx, "missing")
	}
}

// BenchmarkRawMemHit is the baseline for BenchmarkCacheWarmHit: the same
// workload through a raw mem store, which also copies on each Get.
func BenchmarkRawMemHit(b *testing.B) {
	ctx := context.Background()
	inner := mem.New()
	payload := make([]byte, 4096)
	if err := inner.Set(ctx, "k", payload); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := inner.Get(ctx, "k")
		if err != nil {
			b.Fatal(err)
		}
		if len(v) != 4096 {
			b.Fatalf("len %d", len(v))
		}
	}
}

// BenchmarkCacheVsRaw runs the same workload through cache and through raw
// mem to make wrapper overhead visible side-by-side. The two reported numbers
// (Cache and Raw subtests) bracket the wrapper cost.
func BenchmarkCacheVsRaw(b *testing.B) {
	ctx := context.Background()
	inner := mem.New()
	payload := make([]byte, 256)
	if err := inner.Set(ctx, "k", payload); err != nil {
		b.Fatal(err)
	}

	b.Run("Raw", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if _, err := inner.Get(ctx, "k"); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("Cache", func(b *testing.B) {
		c := cache.NewWriter(inner, cache.Config{})
		if _, err := c.Get(ctx, "k"); err != nil {
			b.Fatal(err)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := c.Get(ctx, "k"); err != nil {
				b.Fatal(err)
			}
		}
	})
}
