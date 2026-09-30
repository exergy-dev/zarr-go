// Package cache provides an in-memory LRU caching decorator for store.Store.
//
// New returns a read-only *Store (not a store.Writer, even over a writable
// store), so the zarr write paths reject it with store.ErrReadOnly.
// NewWriter wraps a store.Writer and returns a *Writer, which writes
// through and invalidates.
//
// Concurrent misses on the same key are coalesced into one underlying Get
// (see flight). Unlike golang.org/x/sync's singleflight, each waiter blocks
// on its own context, so a short-deadline caller can't fail out the other
// waiters and a cancelled waiter doesn't wait for the leader. A Set/Delete
// that completes while a fetch is in flight poisons that fetch, so its
// possibly stale result is never admitted, and detaches it, so a Get that
// starts after the write fetches afresh and sees it.
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

	"github.com/exergy-dev/zarr-go/store"
)

// Config configures the cache. The size/count limits are independent and
// may be set to 0 to disable.
type Config struct {
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
// concurrent Get miss on that key until a Set/Delete detaches it.
type flight struct {
	done chan struct{}
	val  []byte
	err  error
	// poisoned: a Set/Delete for this key completed while the fetch was in
	// flight. The result is not admitted and the flight is removed from
	// Store.flights. Waiters that had already joined still receive it,
	// which is linearizable because their Gets overlapped the write.
	poisoned bool
}

// testHookWaiterJoined, when non-nil, is called each time Get joins an
// existing flight as a waiter rather than leading it. Test-only.
var testHookWaiterJoined func()

// testHookFetchPublished, when non-nil, is called by a fetch leader right
// after the locked section that retires its flight (removes it, reads
// poisoned, admits the result), before fl.done is closed. Test-only.
var testHookFetchPublished func()

// New constructs a read-only cache wrapping underlying. The result is not a
// store.Writer even when underlying is one; use NewWriter to keep writes.
//
// Panics if underlying is nil.
func New(underlying store.Store, cfg Config) *Store {
	if underlying == nil {
		panic("cache.New: underlying store is nil")
	}
	return &Store{
		underlying: underlying,
		maxBytes:   cfg.MaxBytes,
		maxEntries: cfg.MaxEntries,
		lru:        list.New(),
		index:      make(map[string]*list.Element),
		flights:    make(map[string]*flight),
	}
}

// lookup returns the cached value for key, if present, WITHOUT copying:
// callers must not mutate it or hand it out uncopied.
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

// admitLocked installs (key,val) into the cache, evicting older entries as
// needed to satisfy MaxBytes / MaxEntries. Caller must hold s.mu. val is
// retained by reference and must not be mutated afterwards. A value larger
// than MaxBytes is not admitted: it could never fit, and admitting it would
// evict every other entry first.
func (s *Store) admitLocked(key string, val []byte) {
	size := int64(len(val))
	if s.maxBytes > 0 && size > s.maxBytes {
		return
	}

	if el, ok := s.index[key]; ok {
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
// stale value after it. The poisoned flight is also detached from s.flights
// so a Get that starts after the mutation begins a fresh fetch instead of
// joining the stale one (read-your-writes). Must be called AFTER the
// underlying mutation completes — see Set/Delete.
func (s *Store) invalidate(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if el, ok := s.index[key]; ok {
		s.removeLocked(el)
	}
	if fl, ok := s.flights[key]; ok {
		fl.poisoned = true
		delete(s.flights, key)
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
// cancellation/deadline error, i.e. the leader's fetch failed because of its
// own ctx rather than a store error. Backends wrap context errors, hence
// errors.Is.
func leaderCtxErr(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// runFetch performs the underlying Get for a flight this goroutine is
// leading, and publishes the result. Always closes fl.done exactly once.
func (s *Store) runFetch(ctx context.Context, key string, fl *flight) {
	raw, err := s.underlying.Get(ctx, key)
	s.mu.Lock()
	if s.flights[key] == fl { // invalidate may have already detached fl and a fresh flight replaced it
		delete(s.flights, key)
	}
	// Admit under the same lock as the poisoned check: a Set/Delete either
	// poisons fl before this point or finds the entry and invalidates it
	// after, so a stale value can't be admitted once the write has completed.
	if err == nil && !fl.poisoned {
		s.admitLocked(key, raw)
	}
	s.mu.Unlock()
	if testHookFetchPublished != nil {
		testHookFetchPublished()
	}

	if err != nil {
		fl.err = err
		close(fl.done)
		return
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

// GetSuffix returns the last n bytes of the value at key. Like GetRange, it
// slices a cached whole value in memory and otherwise delegates to the
// underlying store's GetSuffix without admitting anything.
func (s *Store) GetSuffix(ctx context.Context, key string, n int64) ([]byte, error) {
	if v, ok := s.lookup(key); ok {
		return sliceSuffix(v, n)
	}
	return s.underlying.GetSuffix(ctx, key, n)
}

// sliceRange applies the store.Store GetRange contract to an in-memory
// value.
func sliceRange(v []byte, off, n int64) ([]byte, error) {
	if off < 0 || n < 0 {
		return nil, store.ErrInvalidRange
	}
	return cloneBytes(store.ClampRange(v, off, n)), nil
}

// sliceSuffix applies the store.Store GetSuffix contract to an in-memory
// value.
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

// Exists is a passthrough; existence is not cached.
func (s *Store) Exists(ctx context.Context, key string) (bool, error) {
	return s.underlying.Exists(ctx, key)
}

// List is a passthrough.
func (s *Store) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	return s.underlying.List(ctx, prefix)
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

// Writer is a Store over a store.Writer that also writes through. Every
// write invalidates the key's cached entry and poisons any in-flight fetch
// for it (see flight), so a later Get never returns pre-write bytes.
type Writer struct {
	*Store
	w store.Writer
}

// NewWriter constructs a caching store.Writer wrapping underlying.
//
// Panics if underlying is nil.
func NewWriter(underlying store.Writer, cfg Config) *Writer {
	if underlying == nil {
		panic("cache.NewWriter: underlying store is nil")
	}
	return &Writer{Store: New(underlying, cfg), w: underlying}
}

// Set writes through, then invalidates key. The invalidate happens AFTER
// the underlying write completes (and also poisons, rather than just
// invalidates, any fetch that raced with it) so a concurrent miss-fill
// can't admit a value that's already stale by the time it lands.
func (c *Writer) Set(ctx context.Context, key string, value []byte) error {
	if err := c.w.Set(ctx, key, value); err != nil {
		return err
	}
	c.invalidate(key)
	return nil
}

// Delete forwards to the underlying store, then invalidates key (see Set's
// doc for why this happens after the underlying call).
func (c *Writer) Delete(ctx context.Context, key string) error {
	if err := c.w.Delete(ctx, key); err != nil {
		return err
	}
	c.invalidate(key)
	return nil
}

// SetIfNotExists forwards to the underlying store. key is invalidated
// whether or not the write succeeded, so a subsequent Get re-fetches
// authoritative state.
func (c *Writer) SetIfNotExists(ctx context.Context, key string, value []byte) (bool, error) {
	ok, err := c.w.SetIfNotExists(ctx, key, value)
	c.invalidate(key)
	return ok, err
}

// Compile-time interface assertions.
var (
	_ store.Store  = (*Store)(nil)
	_ store.Writer = (*Writer)(nil)
)
