// Package mem provides an in-memory implementation of store.Store.
//
// It is intended for tests and short-lived caches. All operations are safe for
// concurrent use.
package mem

import (
	"context"
	"fmt"
	"iter"
	"sort"
	"strings"
	"sync"

	"github.com/rkm/zarr-go/store"
)

// Store is an in-memory key/value backend.
//
// The zero value is not usable; construct with New.
type Store struct {
	mu   sync.RWMutex
	data map[string][]byte
}

// New returns an empty in-memory Store.
func New() *Store {
	return &Store{data: make(map[string][]byte)}
}

// Get returns a copy of the value stored at key, or store.ErrKeyNotFound.
func (s *Store) Get(ctx context.Context, key string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.data[key]
	if !ok {
		return nil, store.ErrKeyNotFound
	}
	out := make([]byte, len(v))
	copy(out, v)
	return out, nil
}

// GetRange returns n bytes starting at off from the value at key. The result
// is clamped to whatever bytes lie at or after off; if off >= len(value), an
// empty slice is returned (not an error). Negative off or n is rejected.
//
// The empty-on-off-past-EOF semantics match the cloud backends (s3/gcs/azblob)
// and the fs backend, so codec range-reads see uniform behavior across stores.
func (s *Store) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if off < 0 {
		return nil, fmt.Errorf("mem: GetRange: negative offset %d", off)
	}
	if n < 0 {
		return nil, fmt.Errorf("mem: GetRange: negative length %d", n)
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.data[key]
	if !ok {
		return nil, store.ErrKeyNotFound
	}
	length := int64(len(v))
	if off >= length || n == 0 {
		return []byte{}, nil
	}
	end := off + n
	if end > length || end < off { // clamp + overflow guard
		end = length
	}
	out := make([]byte, end-off)
	copy(out, v[off:end])
	return out, nil
}

// Set stores a defensive copy of value at key.
func (s *Store) Set(ctx context.Context, key string, value []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	cp := make([]byte, len(value))
	copy(cp, value)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data[key] = cp
	return nil
}

// Delete removes key. Missing keys are not an error.
func (s *Store) Delete(ctx context.Context, key string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.data, key)
	return nil
}

// Exists reports whether key is present.
func (s *Store) Exists(ctx context.Context, key string) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.data[key]
	return ok, nil
}

// List yields all keys under prefix in lexicographic order.
func (s *Store) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	s.mu.RLock()
	keys := make([]string, 0, len(s.data))
	for k := range s.data {
		if strings.HasPrefix(k, prefix) {
			keys = append(keys, k)
		}
	}
	s.mu.RUnlock()
	sort.Strings(keys)

	return func(yield func(string, error) bool) {
		if err := ctx.Err(); err != nil {
			yield("", err)
			return
		}
		for _, k := range keys {
			if err := ctx.Err(); err != nil {
				yield("", err)
				return
			}
			if !yield(k, nil) {
				return
			}
		}
	}
}

// SetIfNotExists writes value at key only when key is absent. Returns false
// (and no error) when the key was already present.
func (s *Store) SetIfNotExists(ctx context.Context, key string, value []byte) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.data[key]; ok {
		return false, nil
	}
	cp := make([]byte, len(value))
	copy(cp, value)
	s.data[key] = cp
	return true, nil
}
