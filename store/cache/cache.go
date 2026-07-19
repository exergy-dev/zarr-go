// Package cache provides an in-memory LRU caching decorator for store.Store.
//
// Real workloads (xarray, dashboards, training pipelines) re-read the same
// chunks repeatedly. Wrapping a Store with cache.New eliminates the
// network/disk round-trip on warm hits.
//
// Concurrent misses on the same key are coalesced by a small hand-rolled
// flight tracker (see flight below) so a single underlying Get is performed
// and the result fans out to all waiters. Unlike golang.org/x/sync's
// singleflight, each waiter blocks on its own context (a short-deadline
// caller can't fail out every other waiter, and a waiter's own cancellation
// doesn't block on the leader), and a Set/Delete that completes while a
// fetch is in flight poisons that fetch so its result is never admitted —
// closing a stale-cache window a naive invalidate-then-write has.
//
// GetRange/GetSuffix consult a cached whole value first (sliced in memory)
// before falling back to the underlying store, so a cached shard serves
// every partial-decode range read from memory.
package cache

import (
	"container/list"
	"context"
	"errors"
	"iter"
	"sync"

	"github.com/rkm/zarr-go/store"
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

	mu      sync.Mutex
	lru     *list.List               // front = most recently used
	index   map[string]*list.Element // key -> element holding *entry
	bytes   int64
	flights map[string]*flight // key -> in-progress Get fetch
}

type entry struct {
	key  string
	val  []byte
	size int64
}

// flight tracks one in-flight underlying Get for a key, shared by every
// concurrent Get miss on that key.
type flight struct {
	done     chan struct{}
	val      []byte
	err      error
	poisoned bool // a Set/Delete for this key completed while the fetch was in flight
}

// testHookWaiterJoined, when non-nil, is called each time Get finds an
// existing in-flight fetch for a key and joins it as a waiter (as opposed
// to becoming the leader). Used by tests to deterministically wait for N
// waiters to join a flight before releasing the leader, instead of racing
// on goroutine scheduling. Never set outside tests.
var testHookWaiterJoined func()

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
		flights:    make(map[string]*flight),
	}
}

// lookup returns the cached value for key, if present, WITHOUT copying —
// callers that hand the result to another goroutine or retain it past the
// call must copy it themselves. Internal helper; see Get/GetRange for the
// copying public-facing accessors.
func (s *Store) lookup(key string) ([]byte, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	el, ok := s.index[key]
	if !ok {
		return nil, false
	}
	s.lru.MoveToFront(el)
	return el.Value.(*entry).val, true
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

// invalidate removes any cached entry for key and poisons any in-flight
// fetch for key, so a fill that started before this mutation can't admit a
// stale value after it. Must be called AFTER the underlying mutation
// completes — see Set/Delete.
func (s *Store) invalidate(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if el, ok := s.index[key]; ok {
		s.removeLocked(el)
	}
	if fl, ok := s.flights[key]; ok {
		fl.poisoned = true
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

// Get returns the value at key, consulting the cache first. Concurrent
// misses for the same key are coalesced into a single underlying Get (see
// flight); each waiter blocks on its own ctx, independent of the leader's.
func (s *Store) Get(ctx context.Context, key string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for {
		if v, ok := s.lookup(key); ok {
			return cloneBytes(v), nil
		}

		s.mu.Lock()
		if v, ok := s.index[key]; ok { // admitted between the unlocked lookup and here
			s.lru.MoveToFront(v)
			out := cloneBytes(v.Value.(*entry).val)
			s.mu.Unlock()
			return out, nil
		}
		fl, waiting := s.flights[key] // waiting: someone else is already fetching key
		amLeader := !waiting
		if amLeader {
			fl = &flight{done: make(chan struct{})}
			s.flights[key] = fl
		}
		s.mu.Unlock()
		if waiting && testHookWaiterJoined != nil {
			testHookWaiterJoined()
		}

		if amLeader {
			s.runFetch(ctx, key, fl)
		}

		select {
		case <-fl.done:
			if fl.err != nil {
				if leaderCtxErr(fl.err) && ctx.Err() == nil {
					// The leader's own ctx was cancelled/timed out, not
					// ours; retry — we become the next leader (or join a
					// fresh flight another live waiter started).
					continue
				}
				return nil, fl.err
			}
			return cloneBytes(fl.val), nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

// leaderCtxErr reports whether err is (possibly wrapped) a context
// cancellation/deadline error, i.e. the leader's fetch failed because of ITS
// ctx, not because the underlying store legitimately errored. Cloud SDKs
// typically wrap context errors (e.g. "get object: %w") rather than
// returning the bare sentinel, so this must use errors.Is, not ==.
func leaderCtxErr(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// runFetch performs the underlying Get for a flight this goroutine is
// leading, and publishes the result. Always closes fl.done exactly once.
func (s *Store) runFetch(ctx context.Context, key string, fl *flight) {
	raw, err := s.underlying.Get(ctx, key)
	s.mu.Lock()
	delete(s.flights, key)
	poisoned := fl.poisoned
	s.mu.Unlock()

	if err != nil {
		fl.err = err
		close(fl.done)
		return
	}
	if !poisoned {
		s.admit(key, raw)
	}
	fl.val = raw
	close(fl.done)
}

// cloneBytes returns a defensive copy of b.
func cloneBytes(b []byte) []byte {
	out := make([]byte, len(b))
	copy(out, b)
	return out
}

// GetRange returns n bytes starting at off from the value at key. A cached
// whole value is sliced in memory; otherwise the request is delegated to
// the underlying store's GetRange WITHOUT admitting anything into the cache
// (a range read is not evidence the caller wants the whole value cached).
func (s *Store) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	if v, ok := s.lookup(key); ok {
		return sliceRange(v, off, n)
	}
	return s.underlying.GetRange(ctx, key, off, n)
}

// GetSuffix returns the last n bytes of the value at key, consulting a
// cached whole value first. Falls back to the underlying store's
// GetSuffix if it implements store.SuffixReader, else to a full Get (which
// admits the value, so subsequent range/suffix reads on the same key are
// then served from cache).
func (s *Store) GetSuffix(ctx context.Context, key string, n int64) ([]byte, error) {
	if v, ok := s.lookup(key); ok {
		return sliceSuffix(v, n)
	}
	if n < 0 {
		return nil, store.ErrInvalidRange
	}
	if sr, ok := s.underlying.(store.SuffixReader); ok {
		return sr.GetSuffix(ctx, key, n)
	}
	whole, err := s.Get(ctx, key)
	if err != nil {
		return nil, err
	}
	return sliceSuffix(whole, n)
}

// sliceRange applies the store.Store GetRange contract to an in-memory
// value.
func sliceRange(v []byte, off, n int64) ([]byte, error) {
	if off < 0 || n < 0 {
		return nil, store.ErrInvalidRange
	}
	length := int64(len(v))
	if off >= length || n == 0 {
		return []byte{}, nil
	}
	end := off + n
	if end > length || end < off {
		end = length
	}
	return cloneBytes(v[off:end]), nil
}

// sliceSuffix applies the store.SuffixReader contract to an in-memory value.
func sliceSuffix(v []byte, n int64) ([]byte, error) {
	if n < 0 {
		return nil, store.ErrInvalidRange
	}
	length := int64(len(v))
	start := length - n
	if start < 0 {
		start = 0
	}
	return cloneBytes(v[start:]), nil
}

// Set invalidates any cached entry for key and writes through. The
// invalidate happens AFTER the underlying write completes (and also
// poisons, rather than just invalidates, any fetch that raced with it) so a
// concurrent miss-fill can't admit a value that's already stale by the time
// it lands.
func (s *Store) Set(ctx context.Context, key string, value []byte) error {
	if err := s.underlying.Set(ctx, key, value); err != nil {
		return err
	}
	s.invalidate(key)
	return nil
}

// Delete invalidates/poisons any cached or in-flight entry for key (see
// Set's doc for why this happens after the underlying call) and forwards to
// the underlying store.
func (s *Store) Delete(ctx context.Context, key string) error {
	if err := s.underlying.Delete(ctx, key); err != nil {
		return err
	}
	s.invalidate(key)
	return nil
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
// store.WritableStore. Any cached/in-flight entry for key is
// invalidated/poisoned regardless of whether the write succeeded, so a
// subsequent Get re-fetches authoritative state.
func (s *Store) SetIfNotExists(ctx context.Context, key string, value []byte) (bool, error) {
	w, ok := s.underlying.(store.WritableStore)
	if !ok {
		return false, errNotWritable{}
	}
	ok2, err := w.SetIfNotExists(ctx, key, value)
	s.invalidate(key)
	return ok2, err
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

// Compile-time interface assertions.
var (
	_ store.Store         = (*Store)(nil)
	_ store.WritableStore = (*Store)(nil)
)
