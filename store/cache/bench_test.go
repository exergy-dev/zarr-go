package cache_test

import (
	"context"
	"testing"

	"github.com/rkm/zarr-go/store/cache"
	"github.com/rkm/zarr-go/store/mem"
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
	c := cache.New(cache.Config{Underlying: inner})
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
	c := cache.New(cache.Config{Underlying: inner})
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
// absent in underlying). Singleflight + LRU bookkeeping overhead is what we
// pay for each call.
func BenchmarkCacheMiss(b *testing.B) {
	ctx := context.Background()
	inner := mem.New()
	c := cache.New(cache.Config{Underlying: inner})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = c.Get(ctx, "missing")
	}
}

// BenchmarkRawMemHit is the baseline: same workload through raw mem store.
// Compare against BenchmarkCacheWarmHit -- the cache should be at least as
// fast (it skips a map lookup + slice copy in mem.Store on each call would
// happen anyway, but the cache's defensive copy still applies).
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
		c := cache.New(cache.Config{Underlying: inner})
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
