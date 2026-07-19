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

// StripPrefix removes the configured prefix from a returned object key.
func StripPrefix(prefix, key string) string {
	if prefix == "" {
		return key
	}
	if key == prefix {
		return ""
	}
	return strings.TrimPrefix(key, prefix+"/")
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
