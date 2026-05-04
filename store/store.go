// Package store defines the Store interface that backs all zarr arrays.
//
// Backends in subpackages (fs, s3, mem, ...) implement Store. Everything in
// the rest of the library depends only on this interface.
package store

import (
	"context"
	"errors"
	"io/fs"
	"iter"
	"os"
)

// ErrKeyNotFound is returned by Store.Get/GetRange/Delete/Exists when the key
// is missing. It wraps os.ErrNotExist so errors.Is(err, os.ErrNotExist) works.
var ErrKeyNotFound = errKeyNotFound{}

type errKeyNotFound struct{}

func (errKeyNotFound) Error() string { return "key not found" }
func (errKeyNotFound) Is(target error) bool {
	return target == os.ErrNotExist || target == fs.ErrNotExist || errors.Is(target, errKeyNotFoundType{})
}

type errKeyNotFoundType struct{}

func (errKeyNotFoundType) Error() string { return "key not found" }

// Store is the read/write key-value abstraction over the underlying object/file
// store. Keys are forward-slash separated; backends translate as needed.
type Store interface {
	// Get returns the full value at key. Returns ErrKeyNotFound if absent.
	Get(ctx context.Context, key string) ([]byte, error)

	// GetRange returns n bytes starting at off from the value at key.
	// Returns ErrKeyNotFound if absent. Backends that don't support ranges
	// natively MAY fall back to Get + slice.
	GetRange(ctx context.Context, key string, off, n int64) ([]byte, error)

	// Set writes value at key, replacing any existing value.
	Set(ctx context.Context, key string, value []byte) error

	// Delete removes the value at key. Missing keys are not an error.
	Delete(ctx context.Context, key string) error

	// List yields all keys under prefix. Iteration order is unspecified.
	// Errors during pagination are surfaced via the second tuple value.
	List(ctx context.Context, prefix string) iter.Seq2[string, error]

	// Exists reports whether key is present.
	Exists(ctx context.Context, key string) (bool, error)
}

// WritableStore extends Store with operations needed for safe concurrent
// creation. Not all backends support these (e.g., HTTP read-only). The M2
// write path will require this.
type WritableStore interface {
	Store
	// SetIfNotExists writes value at key only if the key does not already
	// exist. Returns false (without error) if the key was already present.
	SetIfNotExists(ctx context.Context, key string, value []byte) (bool, error)
}
