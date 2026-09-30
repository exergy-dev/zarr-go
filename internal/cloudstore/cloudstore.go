// Package cloudstore holds key-normalization and range-validation helpers
// shared by the s3, gcs, and azblob store backends. Behavior is covered by
// store/storetest's conformance suite via each backend, not by per-package
// duplicate tests.
package cloudstore

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/exergy-dev/zarr-go/store"
)

// ErrEmptyKey is returned (wrapped with the calling backend's name) by
// JoinKey for a key that is empty after stripping leading slashes.
var ErrEmptyKey = errors.New("empty key")

// JoinKey joins the configured prefix with key, producing a clean object key
// with no leading or duplicate slashes, for the single-key operations (Get,
// GetRange, GetSuffix, Set, SetIfNotExists, Delete, Exists).
//
// A key that is empty after stripping leading slashes is rejected with
// ErrEmptyKey: there is no object inside the store's namespace for it to
// name. Joining it would address the object named exactly prefix (outside
// prefix+"/") or, mapped to prefix+"/", the directory marker that
// KeyFromObject hides from List; with no prefix, the empty object name the
// SDKs reject anyway.
func JoinKey(backend, prefix, key string) (string, error) {
	prefix = TrimSlashes(prefix)
	k := strings.TrimLeft(key, "/")
	if k == "" {
		return "", fmt.Errorf("%s: key %q: %w", backend, key, ErrEmptyKey)
	}
	if prefix == "" {
		return k, nil
	}
	return prefix + "/" + k, nil
}

// TrimSlashes strips leading and trailing slashes from s.
func TrimSlashes(s string) string {
	return strings.Trim(s, "/")
}

// ListPrefix returns the object-name prefix to list for a store-level List
// prefix. With a non-empty storePrefix the result always includes the
// separating slash, so listing never matches sibling objects such as
// storePrefix+"2/x" or an object named exactly storePrefix.
func ListPrefix(storePrefix, prefix string) string {
	storePrefix = TrimSlashes(storePrefix)
	prefix = strings.TrimLeft(prefix, "/")
	if storePrefix == "" {
		return prefix
	}
	return storePrefix + "/" + prefix
}

// KeyFromObject maps a listed object name back to a store key. ok is false
// for objects outside storePrefix+"/" and for the bare directory marker
// storePrefix+"/", which has no store key.
func KeyFromObject(storePrefix, objKey string) (key string, ok bool) {
	storePrefix = TrimSlashes(storePrefix)
	if storePrefix == "" {
		return objKey, objKey != ""
	}
	key, ok = strings.CutPrefix(objKey, storePrefix+"/")
	if !ok || key == "" {
		return "", false
	}
	return key, true
}

// ClampLen returns n reduced so that off+n cannot overflow int64, for
// non-negative off and n (call ValidateRange first). Range reads build a
// closed "bytes=off-(off+n-1)" header, and every backend (S3 here, GCS and
// Azure inside their SDKs) computes that end offset unguarded, so an
// unclamped n near MaxInt64 wraps to a negative, malformed end. The clamp
// only shrinks requests ending past MaxInt64, far beyond any real object, so
// the GetRange contract's clamp-to-size result is unchanged. off == MaxInt64
// clamps n to 0, which callers already treat as an existence check.
func ClampLen(off, n int64) int64 {
	return min(n, math.MaxInt64-off)
}

// ValidateRange returns store.ErrInvalidRange (wrapped with the calling
// backend's name, op, and arguments) if off or n is negative, else nil.
// op is typically "GetRange" or "GetSuffix"; off is ignored (pass 0) by
// GetSuffix callers, which have no offset argument.
func ValidateRange(backend, op string, off, n int64) error {
	if off < 0 || n < 0 {
		if op == "GetSuffix" {
			return fmt.Errorf("%s: %s(n=%d): %w", backend, op, n, store.ErrInvalidRange)
		}
		return fmt.Errorf("%s: %s(off=%d, n=%d): %w", backend, op, off, n, store.ErrInvalidRange)
	}
	return nil
}
