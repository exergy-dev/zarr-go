// Package mem provides an in-memory implementation of store.Store.
//
// It is intended for tests and short-lived caches. All operations are safe for
// concurrent use.
package mem

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"strings"
	"sync"

	"github.com/exergy-dev/zarr-go/store"
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

// GetRange returns n bytes starting at off from the value at key, per the
// store.Store GetRange contract: negative off/n is rejected with
// ErrInvalidRange; a missing key is ErrKeyNotFound (checked even when n==0);
// off at or past EOF returns an empty slice; off+n past EOF clamps to
// whatever bytes are available at or after off.
func (s *Store) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if off < 0 || n < 0 {
		return nil, fmt.Errorf("mem: GetRange(off=%d, n=%d): %w", off, n, store.ErrInvalidRange)
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.data[key]
	if !ok {
		return nil, store.ErrKeyNotFound
	}
	clamped := store.ClampRange(v, off, n)
	out := make([]byte, len(clamped))
	copy(out, clamped)
	return out, nil
}

// GetSuffix returns the last n bytes of the value at key, per the
// store.Store GetSuffix contract.
func (s *Store) GetSuffix(ctx context.Context, key string, n int64) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if n < 0 {
		return nil, fmt.Errorf("mem: GetSuffix(n=%d): %w", n, store.ErrInvalidRange)
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.data[key]
	if !ok {
		return nil, store.ErrKeyNotFound
	}
	length := int64(len(v))
	start := length - n
	if start < 0 {
		start = 0
	}
	out := make([]byte, length-start)
	copy(out, v[start:])
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

// List yields all keys under prefix in lexicographic order. Keys are
// snapshotted when the sequence is iterated, not when List is called, so
// (like the fs and cloud backends) each iteration sees the current contents.
func (s *Store) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		if err := ctx.Err(); err != nil {
			yield("", err)
			return
		}
		s.mu.RLock()
		keys := make([]string, 0, len(s.data))
		for k := range s.data {
			if strings.HasPrefix(k, prefix) {
				keys = append(keys, k)
			}
		}
		s.mu.RUnlock()
		slices.Sort(keys)

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

// Compile-time interface assertion.
var _ store.Writer = (*Store)(nil)
