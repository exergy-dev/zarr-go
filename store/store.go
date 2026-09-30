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
)

// ErrKeyNotFound is returned by Store.Get/GetRange/Delete/Exists when the key
// is missing. It wraps os.ErrNotExist so errors.Is(err, os.ErrNotExist) works.
var ErrKeyNotFound = errKeyNotFound{}

// ErrAlreadyExists is returned by higher-level helpers (e.g. CreateArray /
// CreateGroup) when an atomic SetIfNotExists fails because the target key
// already exists. Backends do not return this directly — they signal the
// race by returning (false, nil) from SetIfNotExists; the engine wraps that
// into ErrAlreadyExists at the API surface.
//
// Wraps os.ErrExist so errors.Is(err, os.ErrExist) works for callers that
// already handle that family.
var ErrAlreadyExists = errAlreadyExists{}

type errAlreadyExists struct{}

func (errAlreadyExists) Error() string { return "key already exists" }
func (errAlreadyExists) Is(target error) bool {
	// os.ErrExist and fs.ErrExist are the same value; only one check needed.
	return target == fs.ErrExist
}

type errKeyNotFound struct{}

func (errKeyNotFound) Error() string { return "key not found" }
func (errKeyNotFound) Is(target error) bool {
	// os.ErrNotExist and fs.ErrNotExist are the same value; only one check needed.
	return target == fs.ErrNotExist
}

// ErrInvalidRange is returned by GetRange/GetSuffix when off or n is
// negative. Backends must perform this check client-side, before issuing any
// request — an unvalidated negative offset can silently turn into a
// different (and wrong) request on some backends (e.g. a negative S3 Range
// header offset is reinterpreted as a suffix range by the S3 API, returning
// the whole object instead of erroring).
var ErrInvalidRange = errors.New("store: invalid range")

// ClampRange returns the sub-slice of v covering [off, off+n), clamped to
// len(v) per the GetRange contract: off >= len(v) or n == 0 returns an empty
// slice; off+n > len(v) (including int64 overflow) clamps to len(v). Callers
// are responsible for the off < 0 / n < 0 -> ErrInvalidRange check
// separately, since backends wrap that error differently.
func ClampRange(v []byte, off, n int64) []byte {
	length := int64(len(v))
	if off >= length || n == 0 {
		return []byte{}
	}
	end := off + n
	if end > length || end < off {
		end = length
	}
	return v[off:end]
}

// Store is the read/write key-value abstraction over the underlying object/file
// store. Keys are forward-slash separated; backends translate as needed.
//
// # GetRange contract
//
// All backends implement GetRange identically for the following cases:
//   - off < 0 or n < 0: returns ErrInvalidRange, without issuing any request.
//   - key missing: returns ErrKeyNotFound, even when n == 0 (existence is
//     always checked, whether or not any bytes are requested).
//   - n == 0 and key exists: returns ([]byte{}, nil).
//   - off >= size (key exists): returns ([]byte{}, nil), not an error.
//   - off+n > size: clamped to return size-off bytes (the largest slice
//     available at or after off).
//
// # List contract
//
// List(ctx, prefix) yields every key for which strings.HasPrefix(key,
// prefix) holds, recursively, in unspecified order. It never yields keys
// that aren't logically part of the stored keyspace (e.g. a backend's own
// temp/staging artifacts).
type Store interface {
	// Get returns the full value at key. Returns ErrKeyNotFound if absent.
	Get(ctx context.Context, key string) ([]byte, error)

	// GetRange returns n bytes starting at off from the value at key. See
	// the Store-level "GetRange contract" doc for the full behavior matrix.
	GetRange(ctx context.Context, key string, off, n int64) ([]byte, error)

	// Set writes value at key, replacing any existing value.
	Set(ctx context.Context, key string, value []byte) error

	// Delete removes the value at key. Missing keys are not an error.
	Delete(ctx context.Context, key string) error

	// List yields all keys under prefix. See the Store-level "List contract"
	// doc. Errors during pagination are surfaced via the second tuple value.
	List(ctx context.Context, prefix string) iter.Seq2[string, error]

	// Exists reports whether key is present.
	Exists(ctx context.Context, key string) (bool, error)
}

// SuffixReader is an optional Store extension implemented by backends that
// can serve "last N bytes" requests — natively (HTTP Range: bytes=-N-style
// backends) or otherwise. Consumers that need this (e.g. the sharding codec,
// reading a shard's index trailer) probe for it via a type assertion and
// fall back to Get + slice when a backend doesn't implement it.
//
// GetSuffix follows the same off/n conventions as GetRange with n playing
// the role of the suffix length: n < 0 returns ErrInvalidRange; n == 0
// returns ([]byte{}, nil) if the key exists (ErrKeyNotFound if not); n >=
// size returns the whole value; otherwise the last n bytes.
type SuffixReader interface {
	GetSuffix(ctx context.Context, key string, n int64) ([]byte, error)
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

// MmapStore is implemented by Stores that can return chunk bytes via memory
// mapping. The returned slice is valid only until release() is called; the
// engine must call release() when it's done with the bytes (typically right
// after pipeline.Decode finishes).
//
// Callers that don't probe for MmapStore continue using Get() unchanged.
// release is always non-nil, even on error paths where it's a no-op, so callers
// can defer it unconditionally after a successful call.
type MmapStore interface {
	GetMmap(ctx context.Context, key string) (data []byte, release func(), err error)
}
