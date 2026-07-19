package cache_test

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"sync/atomic"
	"testing"

	"github.com/rkm/zarr-go/store"
	"github.com/rkm/zarr-go/store/cache"
	"github.com/rkm/zarr-go/store/mem"
	"github.com/rkm/zarr-go/store/storetest"
)

// countingStore wraps a store.Store and records call counts per method for
// assertions.
type countingStore struct {
	inner store.Store

	getCalls    atomic.Int64
	rangeCalls  atomic.Int64
	setCalls    atomic.Int64
	deleteCalls atomic.Int64
	existsCalls atomic.Int64
	listCalls   atomic.Int64
}

func newCounting(inner store.Store) *countingStore {
	return &countingStore{inner: inner}
}

func (c *countingStore) Get(ctx context.Context, key string) ([]byte, error) {
	c.getCalls.Add(1)
	return c.inner.Get(ctx, key)
}

func (c *countingStore) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	c.rangeCalls.Add(1)
	return c.inner.GetRange(ctx, key, off, n)
}

func (c *countingStore) Set(ctx context.Context, key string, value []byte) error {
	c.setCalls.Add(1)
	return c.inner.Set(ctx, key, value)
}

func (c *countingStore) Delete(ctx context.Context, key string) error {
	c.deleteCalls.Add(1)
	return c.inner.Delete(ctx, key)
}

func (c *countingStore) Exists(ctx context.Context, key string) (bool, error) {
	c.existsCalls.Add(1)
	return c.inner.Exists(ctx, key)
}

func (c *countingStore) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	c.listCalls.Add(1)
	return c.inner.List(ctx, prefix)
}

func TestCacheHitReturnsCachedBytes(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	inner := newCounting(mem.New())
	if err := inner.inner.Set(ctx, "k", []byte("v")); err != nil {
		t.Fatal(err)
	}
	c := cache.New(cache.Config{Underlying: inner})

	v1, err := c.Get(ctx, "k")
	if err != nil {
		t.Fatalf("first Get: %v", err)
	}
	if string(v1) != "v" {
		t.Fatalf("first Get: got %q want %q", v1, "v")
	}
	v2, err := c.Get(ctx, "k")
	if err != nil {
		t.Fatalf("second Get: %v", err)
	}
	if string(v2) != "v" {
		t.Fatalf("second Get: got %q want %q", v2, "v")
	}
	if got := inner.getCalls.Load(); got != 1 {
		t.Fatalf("underlying Get calls: got %d want 1", got)
	}
}

func TestCacheEvictionByCount(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	inner := mem.New()
	for _, k := range []string{"a", "b", "c"} {
		if err := inner.Set(ctx, k, []byte(k)); err != nil {
			t.Fatal(err)
		}
	}
	c := cache.New(cache.Config{Underlying: inner, MaxEntries: 2})

	for _, k := range []string{"a", "b", "c"} {
		if _, err := c.Get(ctx, k); err != nil {
			t.Fatalf("Get %s: %v", k, err)
		}
	}
	if got := c.Len(); got != 2 {
		t.Fatalf("Len: got %d want 2", got)
	}
	// "a" is the oldest and should have been evicted.
	counting := newCounting(inner)
	c2 := cache.New(cache.Config{Underlying: counting, MaxEntries: 2})
	for _, k := range []string{"a", "b", "c"} {
		if _, err := c2.Get(ctx, k); err != nil {
			t.Fatal(err)
		}
	}
	// Now Get "a" again -- should miss.
	if _, err := c2.Get(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	// Initial 3 + 1 refetch for "a".
	if got := counting.getCalls.Load(); got != 4 {
		t.Fatalf("underlying Get calls: got %d want 4", got)
	}
}

func TestCacheEvictionBySize(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	inner := mem.New()
	// Eight entries of 25 bytes each = 200 bytes total.
	payload := make([]byte, 25)
	for i := range payload {
		payload[i] = 'x'
	}
	for i := 0; i < 8; i++ {
		if err := inner.Set(ctx, fmt.Sprintf("k%d", i), payload); err != nil {
			t.Fatal(err)
		}
	}
	c := cache.New(cache.Config{Underlying: inner, MaxBytes: 100})
	for i := 0; i < 8; i++ {
		if _, err := c.Get(ctx, fmt.Sprintf("k%d", i)); err != nil {
			t.Fatal(err)
		}
	}
	if got := c.Bytes(); got > 100 {
		t.Fatalf("Bytes: got %d want <= 100", got)
	}
	// We expect exactly 4 entries cached (4 * 25 = 100).
	if got := c.Len(); got != 4 {
		t.Fatalf("Len: got %d want 4", got)
	}
}

func TestCacheInvalidatesOnSet(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	inner := mem.New()
	if err := inner.Set(ctx, "k", []byte("old")); err != nil {
		t.Fatal(err)
	}
	c := cache.New(cache.Config{Underlying: inner})

	if v, err := c.Get(ctx, "k"); err != nil || string(v) != "old" {
		t.Fatalf("Get: %q %v", v, err)
	}
	if err := c.Set(ctx, "k", []byte("new")); err != nil {
		t.Fatal(err)
	}
	v, err := c.Get(ctx, "k")
	if err != nil {
		t.Fatal(err)
	}
	if string(v) != "new" {
		t.Fatalf("Get after Set: got %q want %q", v, "new")
	}
}

func TestCacheInvalidatesOnDelete(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	inner := mem.New()
	if err := inner.Set(ctx, "k", []byte("v")); err != nil {
		t.Fatal(err)
	}
	c := cache.New(cache.Config{Underlying: inner})

	if _, err := c.Get(ctx, "k"); err != nil {
		t.Fatal(err)
	}
	if err := c.Delete(ctx, "k"); err != nil {
		t.Fatal(err)
	}
	_, err := c.Get(ctx, "k")
	if !errors.Is(err, store.ErrKeyNotFound) {
		t.Fatalf("expected ErrKeyNotFound, got %v", err)
	}
}

func TestCacheRangePassthrough(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	inner := newCounting(mem.New())
	if err := inner.inner.Set(ctx, "k", []byte("hello world")); err != nil {
		t.Fatal(err)
	}
	c := cache.New(cache.Config{Underlying: inner})

	for i := 0; i < 3; i++ {
		v, err := c.GetRange(ctx, "k", 0, 5)
		if err != nil {
			t.Fatal(err)
		}
		if string(v) != "hello" {
			t.Fatalf("got %q want %q", v, "hello")
		}
	}
	if got := inner.rangeCalls.Load(); got != 3 {
		t.Fatalf("rangeCalls: got %d want 3", got)
	}
	if got := inner.getCalls.Load(); got != 0 {
		t.Fatalf("getCalls: got %d want 0 (range must not promote)", got)
	}
}

func TestCacheReturnsDefensiveCopy(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	inner := mem.New()
	if err := inner.Set(ctx, "k", []byte("hello")); err != nil {
		t.Fatal(err)
	}
	c := cache.New(cache.Config{Underlying: inner})

	v1, err := c.Get(ctx, "k")
	if err != nil {
		t.Fatal(err)
	}
	for i := range v1 {
		v1[i] = 'X'
	}
	v2, err := c.Get(ctx, "k")
	if err != nil {
		t.Fatal(err)
	}
	if string(v2) != "hello" {
		t.Fatalf("cache corrupted: got %q want %q", v2, "hello")
	}
}

func TestCachePropagatesNotFound(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	c := cache.New(cache.Config{Underlying: mem.New()})

	_, err := c.Get(ctx, "missing")
	if !errors.Is(err, store.ErrKeyNotFound) {
		t.Fatalf("expected ErrKeyNotFound, got %v", err)
	}
}

func TestCacheExistsAndListPassthrough(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	inner := newCounting(mem.New())
	if err := inner.inner.Set(ctx, "a/1", []byte("x")); err != nil {
		t.Fatal(err)
	}
	c := cache.New(cache.Config{Underlying: inner})

	ok, err := c.Exists(ctx, "a/1")
	if err != nil || !ok {
		t.Fatalf("Exists: %v %v", ok, err)
	}
	if got := inner.existsCalls.Load(); got != 1 {
		t.Fatalf("existsCalls: got %d want 1", got)
	}

	count := 0
	for k, err := range c.List(ctx, "a/") {
		if err != nil {
			t.Fatal(err)
		}
		if k != "a/1" {
			t.Fatalf("listed %q", k)
		}
		count++
	}
	if count != 1 {
		t.Fatalf("List yielded %d items", count)
	}
}

func TestCacheZeroLimitsDisable(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	inner := mem.New()
	for i := 0; i < 1000; i++ {
		if err := inner.Set(ctx, fmt.Sprintf("k%d", i), []byte("v")); err != nil {
			t.Fatal(err)
		}
	}
	c := cache.New(cache.Config{Underlying: inner})
	for i := 0; i < 1000; i++ {
		if _, err := c.Get(ctx, fmt.Sprintf("k%d", i)); err != nil {
			t.Fatal(err)
		}
	}
	if got := c.Len(); got != 1000 {
		t.Fatalf("Len: got %d want 1000 (no limit means no eviction)", got)
	}
}

func TestCacheLRUOrdering(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	inner := mem.New()
	for _, k := range []string{"a", "b", "c"} {
		if err := inner.Set(ctx, k, []byte(k)); err != nil {
			t.Fatal(err)
		}
	}
	c := cache.New(cache.Config{Underlying: inner, MaxEntries: 2})

	if _, err := c.Get(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Get(ctx, "b"); err != nil {
		t.Fatal(err)
	}
	// Touch "a" to make it most-recent.
	if _, err := c.Get(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	// Now load "c": this should evict "b", not "a".
	if _, err := c.Get(ctx, "c"); err != nil {
		t.Fatal(err)
	}

	// "b" should be the evicted one. Verify via a fresh counting wrapper
	// over a brand-new cache that mirrors the same access pattern.
	cnt := newCounting(inner)
	c2 := cache.New(cache.Config{Underlying: cnt, MaxEntries: 2})
	for _, k := range []string{"a", "b"} {
		if _, err := c2.Get(ctx, k); err != nil {
			t.Fatal(err)
		}
	}
	// Touch "a" to keep it.
	if _, err := c2.Get(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	if _, err := c2.Get(ctx, "c"); err != nil {
		t.Fatal(err)
	}
	beforeA := cnt.getCalls.Load()
	if _, err := c2.Get(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	if cnt.getCalls.Load() != beforeA {
		t.Fatalf("'a' should still be cached; underlying Get fired")
	}
	beforeB := cnt.getCalls.Load()
	if _, err := c2.Get(ctx, "b"); err != nil {
		t.Fatal(err)
	}
	if cnt.getCalls.Load() != beforeB+1 {
		t.Fatalf("'b' should have been evicted; expected refetch")
	}
}

func TestCacheGetRangeServesFromCachedValue(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	inner := newCounting(mem.New())
	if err := inner.inner.Set(ctx, "k", []byte("hello world")); err != nil {
		t.Fatal(err)
	}
	c := cache.New(cache.Config{Underlying: inner})

	if _, err := c.Get(ctx, "k"); err != nil {
		t.Fatal(err)
	}
	v, err := c.GetRange(ctx, "k", 6, 5)
	if err != nil {
		t.Fatal(err)
	}
	if string(v) != "world" {
		t.Fatalf("GetRange = %q, want %q", v, "world")
	}
	if got := inner.rangeCalls.Load(); got != 0 {
		t.Fatalf("rangeCalls: got %d want 0 (should serve from the cached whole value)", got)
	}
}

// suffixCountingStore extends countingStore with a store.SuffixReader
// implementation, so tests can control whether the "underlying implements
// SuffixReader" path is exercised.
type suffixCountingStore struct {
	*countingStore
	suffixCalls atomic.Int64
}

func (c *suffixCountingStore) GetSuffix(ctx context.Context, key string, n int64) ([]byte, error) {
	c.suffixCalls.Add(1)
	sr, ok := c.inner.(store.SuffixReader)
	if !ok {
		return nil, errors.New("inner does not implement store.SuffixReader")
	}
	return sr.GetSuffix(ctx, key, n)
}

func TestCacheGetSuffixServesFromCachedValue(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	inner := newCounting(mem.New())
	if err := inner.inner.Set(ctx, "k", []byte("hello world")); err != nil {
		t.Fatal(err)
	}
	c := cache.New(cache.Config{Underlying: inner})

	if _, err := c.Get(ctx, "k"); err != nil {
		t.Fatal(err)
	}
	v, err := c.GetSuffix(ctx, "k", 5)
	if err != nil {
		t.Fatal(err)
	}
	if string(v) != "world" {
		t.Fatalf("GetSuffix = %q, want %q", v, "world")
	}
}

func TestCacheGetSuffixDelegatesToUnderlyingSuffixReader(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	base := newCounting(mem.New())
	if err := base.inner.Set(ctx, "k", []byte("hello world")); err != nil {
		t.Fatal(err)
	}
	inner := &suffixCountingStore{countingStore: base}
	c := cache.New(cache.Config{Underlying: inner})

	// Nothing cached yet: must delegate to the underlying GetSuffix, not
	// fall back to a whole-blob Get.
	v, err := c.GetSuffix(ctx, "k", 5)
	if err != nil {
		t.Fatal(err)
	}
	if string(v) != "world" {
		t.Fatalf("GetSuffix = %q, want %q", v, "world")
	}
	if got := inner.suffixCalls.Load(); got != 1 {
		t.Fatalf("suffixCalls: got %d want 1", got)
	}
	if got := inner.getCalls.Load(); got != 0 {
		t.Fatalf("getCalls: got %d want 0 (must not fall back to whole Get when SuffixReader is available)", got)
	}
}

func TestCacheGetSuffixFallsBackToGetWhenUnderlyingLacksSuffixReader(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	inner := newCounting(mem.New()) // countingStore does not implement store.SuffixReader
	if err := inner.inner.Set(ctx, "k", []byte("hello world")); err != nil {
		t.Fatal(err)
	}
	c := cache.New(cache.Config{Underlying: inner})

	v, err := c.GetSuffix(ctx, "k", 5)
	if err != nil {
		t.Fatal(err)
	}
	if string(v) != "world" {
		t.Fatalf("GetSuffix = %q, want %q", v, "world")
	}
	if got := inner.getCalls.Load(); got != 1 {
		t.Fatalf("getCalls: got %d want 1 (fallback to whole Get)", got)
	}
	// The fallback Get admits the value, so a second GetSuffix on the same
	// key should now be served from the cache.
	if _, err := c.GetSuffix(ctx, "k", 5); err != nil {
		t.Fatal(err)
	}
	if got := inner.getCalls.Load(); got != 1 {
		t.Fatalf("getCalls after second GetSuffix: got %d want 1 (should serve from cache)", got)
	}
}

func TestCacheSetIfNotExists(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	c := cache.New(cache.Config{Underlying: mem.New()})

	ok1, err := c.SetIfNotExists(ctx, "k", []byte("first"))
	if err != nil || !ok1 {
		t.Fatalf("first SetIfNotExists: ok=%v err=%v", ok1, err)
	}
	// Populate the cache so we can verify it gets invalidated by the second
	// (losing) SetIfNotExists call, not left stale.
	if v, err := c.Get(ctx, "k"); err != nil || string(v) != "first" {
		t.Fatalf("Get: %q %v", v, err)
	}

	ok2, err := c.SetIfNotExists(ctx, "k", []byte("second"))
	if err != nil || ok2 {
		t.Fatalf("second SetIfNotExists: ok=%v err=%v, want false,nil", ok2, err)
	}
	got, err := c.Get(ctx, "k")
	if err != nil || string(got) != "first" {
		t.Fatalf("Get after CAS: got %q err=%v, want %q,nil", got, err, "first")
	}
}

func TestCacheSetIfNotExistsNotWritable(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	c := cache.New(cache.Config{Underlying: readOnlyStore{mem.New()}})
	if _, err := c.SetIfNotExists(ctx, "k", []byte("v")); err == nil {
		t.Fatal("expected error when underlying store isn't WritableStore")
	}
}

// readOnlyStore wraps a store.Store but deliberately does NOT implement
// store.WritableStore, so a type assertion for it fails as intended.
type readOnlyStore struct {
	store.Store
}

func TestConformance(t *testing.T) {
	storetest.RunConformance(t, func(t *testing.T) store.Store {
		return cache.New(cache.Config{Underlying: mem.New()})
	}, storetest.Options{})
}
