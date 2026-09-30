package s3

// Key and range helpers. The gcs and azblob modules carry their own copies:
// each backend is a separate module, and these few lines are cheaper to
// duplicate than a shared module every backend would have to version with.
// Keep the three copies in sync.

import (
	"fmt"
	"math"
	"strings"

	zstore "github.com/exergy-dev/zarr-go/store"
)

// joinKey joins the store's prefix with key into an object name with no
// leading or duplicate slashes, for the single-key operations (Get,
// GetRange, GetSuffix, Set, SetIfNotExists, Delete, Exists).
//
// A key that is empty after stripping leading slashes fails with
// store.ErrInvalidKey: it names no object inside the store's namespace.
// Joining it would address the object named exactly prefix (outside
// prefix+"/") or, mapped to prefix+"/", the directory marker that
// keyFromObject hides from List; with no prefix, the empty object name the
// SDK rejects anyway.
func joinKey(prefix, key string) (string, error) {
	prefix = strings.Trim(prefix, "/")
	k := strings.TrimLeft(key, "/")
	if k == "" {
		return "", fmt.Errorf("s3: empty key %q: %w", key, zstore.ErrInvalidKey)
	}
	if prefix == "" {
		return k, nil
	}
	return prefix + "/" + k, nil
}

// listPrefix returns the object-name prefix to list for a List prefix. With
// a non-empty storePrefix the result always includes the separating slash,
// so listing never matches sibling objects such as storePrefix+"2/x" or an
// object named exactly storePrefix.
func listPrefix(storePrefix, prefix string) string {
	storePrefix = strings.Trim(storePrefix, "/")
	prefix = strings.TrimLeft(prefix, "/")
	if storePrefix == "" {
		return prefix
	}
	return storePrefix + "/" + prefix
}

// keyFromObject maps a listed object name back to a store key. ok is false
// for objects outside storePrefix+"/" and for the bare directory marker
// storePrefix+"/", which has no store key.
func keyFromObject(storePrefix, objKey string) (key string, ok bool) {
	storePrefix = strings.Trim(storePrefix, "/")
	if storePrefix == "" {
		return objKey, objKey != ""
	}
	key, ok = strings.CutPrefix(objKey, storePrefix+"/")
	if !ok || key == "" {
		return "", false
	}
	return key, true
}

// clampLen returns n reduced so that off+n cannot overflow int64, for
// non-negative off and n (check them with checkRange first). A range read
// becomes a closed "bytes=off-(off+n-1)" header, built here or inside the
// SDK without an overflow check, so an unclamped n near MaxInt64 would wrap
// the end offset negative. The clamp only shrinks requests ending past
// MaxInt64, far beyond any real object, so the GetRange contract's
// clamp-to-size result is unchanged. off == MaxInt64 clamps n to 0, which
// GetRange treats as an existence check.
func clampLen(off, n int64) int64 {
	return min(n, math.MaxInt64-off)
}

// checkRange returns store.ErrInvalidRange if GetRange's off or n is
// negative.
func checkRange(off, n int64) error {
	if off < 0 || n < 0 {
		return fmt.Errorf("s3: GetRange(off=%d, n=%d): %w", off, n, zstore.ErrInvalidRange)
	}
	return nil
}

// checkSuffix returns store.ErrInvalidRange if GetSuffix's n is negative.
func checkSuffix(n int64) error {
	if n < 0 {
		return fmt.Errorf("s3: GetSuffix(n=%d): %w", n, zstore.ErrInvalidRange)
	}
	return nil
}
