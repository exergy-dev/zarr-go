// Package cloudstore holds key-normalization and range-validation helpers
// shared by the s3, gcs, and azblob store backends. Behavior is covered by
// store/storetest's conformance suite via each backend, not by per-package
// duplicate tests.
package cloudstore

import (
	"fmt"
	"strings"

	"github.com/rkm/zarr-go/store"
)

// JoinKey joins the configured prefix with key, producing a clean object key
// with no leading or duplicate slashes.
func JoinKey(prefix, key string) string {
	prefix = TrimSlashes(prefix)
	key = strings.TrimLeft(key, "/")
	if prefix == "" {
		return key
	}
	if key == "" {
		return prefix
	}
	return prefix + "/" + key
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
