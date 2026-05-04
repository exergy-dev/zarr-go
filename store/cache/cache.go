// Package cache provides an in-memory LRU caching decorator for store.Store.
//
// Real workloads (xarray, dashboards, training pipelines) re-read the same
// chunks repeatedly. Wrapping a Store with cache.New eliminates the
// network/disk round-trip on warm hits.
//
// Concurrent misses on the same key are coalesced via singleflight so a single
// underlying Get is performed and the result fan-outs to all waiters.
//
// GetRange requests are passed through unchanged: range reads are typically
// sharding-index trailers where each call uses a different (off, n), so
// caching them by key alone would either be unsafe or wasteful.
package cache

import (
	"container/list"
	"context"
	"iter"
	"sync"

	"github.com/rkm/zarr-go/store"
	"golang.org/x/sync/singleflight"
)

// Config configures the cache. Underlying is required; the size/count limits
// are independent and may be set to 0 to disable.
type Config struct {
	// Underlying is the wrapped store. Required.
	Underlying store.Store
	// MaxBytes caps total cached payload bytes. 0 disables the byte limit.
	MaxBytes int64
	// MaxEntries caps the number of cached entries. 0 disables the count limit.
	MaxEntries int
}

// Store wraps an underlying store.Store with an in-memory LRU cache.
//
// Thread-safe. Cache entries hold defensive copies; callers are free to
// mutate the slice returned by Get without affecting cached state.
type Store struct {
	underlying store.Store
	maxBytes   int64
	maxEntries int

	mu    sync.Mutex
	lru   *list.List               // front = most recently used
	index map[string]*list.Element // key -> element holding *entry
	bytes int64

	sf singleflight.Group
}

type entry struct {
	key  string
	val  []byte
	size int64
}

// New constructs a cache wrapping cfg.Underlying.
//
// Panics if cfg.Underlying is nil.
func New(cfg Config) *Store {
	if cfg.Underlying == nil {
		panic("cache.New: cfg.Underlying is nil")
	}
	return &Store{
		underlying: cfg.Underlying,
		maxBytes:   cfg.MaxBytes,
		maxEntries: cfg.MaxEntries,
		lru:        list.New(),
		index:      make(map[string]*list.Element),
	}
}

// lookup returns a defensive copy of the cached value for key, if present.
func (s *Store) lookup(key string) ([]byte, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	el, ok := s.index[key]
	if !ok {
		return nil, false
	}
	s.lru.MoveToFront(el)
	e := el.Value.(*entry)
	out := make([]byte, len(e.val))
	copy(out, e.val)
	return out, true
}

// admit installs (key,val) into the cache, evicting older entries as needed
// to satisfy MaxBytes / MaxEntries. The provided val is taken by reference
// and MUST NOT be mutated by the caller after admission; callers should pass
// freshly-allocated buffers.
func (s *Store) admit(key string, val []byte) {
	size := int64(len(val))
	s.mu.Lock()
	defer s.mu.Unlock()

	if el, ok := s.index[key]; ok {
		// Replace existing.
		old := el.Value.(*entry)
		s.bytes -= old.size
		old.val = val
		old.size = size
		s.bytes += size
		s.lru.MoveToFront(el)
	} else {
		e := &entry{key: key, val: val, size: size}
		el := s.lru.PushFront(e)
		s.index[key] = el
		s.bytes += size
	}
	s.evictLocked()
}

// invalidate removes any cached entry for key.
func (s *Store) invalidate(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if el, ok := s.index[key]; ok {
		s.removeLocked(el)
	}
}

func (s *Store) removeLocked(el *list.Element) {
	e := el.Value.(*entry)
	s.bytes -= e.size
	delete(s.index, e.key)
	s.lru.Remove(el)
}

// evictLocked drops LRU-tail entries until both limits are respected.
func (s *Store) evictLocked() {
	for {
		over := false
		if s.maxEntries > 0 && s.lru.Len() > s.maxEntries {
			over = true
		}
		if s.maxBytes > 0 && s.bytes > s.maxBytes {
			over = true
		}
		if !over {
			return
		}
		tail := s.lru.Back()
		if tail == nil {
			return
		}
		s.removeLocked(tail)
	}
}

// Get returns the value at key, consulting the cache first. Concurrent misses
// for the same key are coalesced into a single underlying Get.
func (s *Store) Get(ctx context.Context, key string) ([]byte, error) {
	if v, ok := s.lookup(key); ok {
		return v, nil
	}
	// Coalesce concurrent misses. The singleflight result holds the freshly
	// fetched payload; we admit a reference into the cache and hand callers
	// independent defensive copies.
	v, err, _ := s.sf.Do(key, func() (any, error) {
		// Re-check after winning the flight in case another goroutine
		// admitted the key between our miss and acquiring leadership.
		if cached, ok := s.lookup(key); ok {
			return cached, nil
		}
		raw, err := s.underlying.Get(ctx, key)
		if err != nil {
			return nil, err
		}
		// Admit a copy so the cache owns its bytes independent of whatever
		// the caller does with the returned slice.
		stored := make([]byte, len(raw))
		copy(stored, raw)
		s.admit(key, stored)
		return raw, nil
	})
	if err != nil {
		return nil, err
	}
	src := v.([]byte)
	out := make([]byte, len(src))
	copy(out, src)
	return out, nil
}

// GetRange passes through to the underlying store. Range reads are not
// cached.
func (s *Store) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	return s.underlying.GetRange(ctx, key, off, n)
}

// Set invalidates any cached entry for key and writes through. Returns an
// error if the underlying store does not implement store.WritableStore-style
// write semantics (it must satisfy store.Store, which includes Set).
func (s *Store) Set(ctx context.Context, key string, value []byte) error {
	s.invalidate(key)
	return s.underlying.Set(ctx, key, value)
}

// Delete invalidates any cached entry for key and forwards to the underlying
// store.
func (s *Store) Delete(ctx context.Context, key string) error {
	s.invalidate(key)
	return s.underlying.Delete(ctx, key)
}

// Exists is a passthrough; existence is not cached.
func (s *Store) Exists(ctx context.Context, key string) (bool, error) {
	return s.underlying.Exists(ctx, key)
}

// List is a passthrough.
func (s *Store) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	return s.underlying.List(ctx, prefix)
}

// SetIfNotExists forwards to the underlying store when it implements
// store.WritableStore. The cache entry for key is invalidated regardless of
// whether the write succeeded, so a subsequent Get re-fetches authoritative
// state.
func (s *Store) SetIfNotExists(ctx context.Context, key string, value []byte) (bool, error) {
	w, ok := s.underlying.(store.WritableStore)
	if !ok {
		return false, errNotWritable{}
	}
	s.invalidate(key)
	return w.SetIfNotExists(ctx, key, value)
}

type errNotWritable struct{}

func (errNotWritable) Error() string {
	return "cache: underlying store does not implement store.WritableStore"
}

// Len reports the current number of cached entries. Intended for tests and
// metrics.
func (s *Store) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lru.Len()
}

// Bytes reports the current cached byte total. Intended for tests and metrics.
func (s *Store) Bytes() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.bytes
}
