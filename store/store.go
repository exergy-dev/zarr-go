// Package store defines the Store interface that backs all zarr arrays.
//
// Backends in subpackages (fs, s3, mem, ...) implement Store, and Writer when
// they can be written. The rest of the library depends only on these
// interfaces.
package store

import (
	"context"
	"errors"
	"io/fs"
	"iter"
)

// ErrKeyNotFound is returned by Get, GetRange, GetSuffix and GetMmap when the
// key is missing. errors.Is(err, os.ErrNotExist) also matches it.
var ErrKeyNotFound = errKeyNotFound{}

// ErrAlreadyExists is returned by CreateArray / CreateGroup when the target
// already exists. Backends do not return it: SetIfNotExists reports the
// conflict as (false, nil). errors.Is(err, os.ErrExist) also matches it.
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
// negative. Backends must check this client-side, before issuing any
// request: some backends reinterpret a negative offset as a different
// request (S3 treats a negative Range offset as a suffix range).
var ErrInvalidRange = errors.New("store: invalid range")

// ErrInvalidKey is returned (wrapped with the backend and key) for a key the
// backend refuses to address: an empty key on the cloud backends, a key
// with a ".." segment on the fs and http stores, or one the fs store can't
// map to a path beneath its root. The check is client-side; no request or
// file operation is issued.
var ErrInvalidKey = errors.New("store: invalid key")

// ErrReadOnly reports a write attempted through a Store that is not a
// Writer. The zarr write paths return it; zarr.ErrReadOnly is the same
// value.
var ErrReadOnly = errors.New("store is read-only")

// ClampRange returns the sub-slice of v covering [off, off+n), clamped to
// len(v) per the GetRange contract: off >= len(v) or n == 0 returns an empty
// slice; off+n > len(v) (including int64 overflow) clamps to len(v). Callers
// check off < 0 / n < 0 themselves, since backends wrap ErrInvalidRange
// differently.
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

// Store is the read side of the key-value abstraction over the underlying
// object/file store. Keys are forward-slash separated; backends translate as
// needed. A Store that can also be written implements Writer; one that
// can't (e.g. the http store) is read-only, and the zarr write paths reject
// it with ErrReadOnly.
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
// # GetSuffix contract
//
// GetSuffix follows the same conventions, with n as the suffix length:
//   - n < 0: returns ErrInvalidRange, without issuing any request.
//   - key missing: returns ErrKeyNotFound, even when n == 0.
//   - n == 0 and key exists: returns ([]byte{}, nil).
//   - n >= size: returns the whole value.
//   - otherwise: returns the last n bytes.
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
	// "GetRange contract" above.
	GetRange(ctx context.Context, key string, off, n int64) ([]byte, error)

	// GetSuffix returns the last n bytes of the value at key without
	// knowing its size up front (the sharding codec reads a shard's index
	// trailer this way). See "GetSuffix contract" above.
	GetSuffix(ctx context.Context, key string, n int64) ([]byte, error)

	// List yields all keys under prefix. See "List contract" above. Errors
	// (e.g. during pagination) are yielded as the second value.
	List(ctx context.Context, prefix string) iter.Seq2[string, error]

	// Exists reports whether key is present.
	Exists(ctx context.Context, key string) (bool, error)
}

// Writer is a Store that can also be written. The zarr write paths
// (CreateArray, CreateGroup, Array.Write) require one and return
// ErrReadOnly for a Store that isn't.
type Writer interface {
	Store

	// Set writes value at key, replacing any existing value.
	Set(ctx context.Context, key string, value []byte) error

	// Delete removes the value at key. Missing keys are not an error.
	Delete(ctx context.Context, key string) error

	// SetIfNotExists writes value at key only if the key does not already
	// exist. Returns false (without error) if the key was already present.
	SetIfNotExists(ctx context.Context, key string, value []byte) (bool, error)
}

// MmapStore is implemented by Stores that can return chunk bytes via memory
// mapping. The returned slice is valid only until release is called, which
// the caller must do once it is done with the bytes. release is always
// non-nil, even on error.
type MmapStore interface {
	GetMmap(ctx context.Context, key string) (data []byte, release func(), err error)
}
