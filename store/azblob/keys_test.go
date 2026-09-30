package azblob

import (
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/exergy-dev/zarr-go/store"
)

func TestJoinKey(t *testing.T) {
	cases := []struct {
		prefix, key, want string
	}{
		{"", "foo", "foo"},
		{"", "/foo", "foo"},
		{"p", "foo", "p/foo"},
		{"p/", "foo", "p/foo"},
		{"/p/", "foo", "p/foo"},
		{"/p/", "/foo", "p/foo"},
		{"a/b", "c/d", "a/b/c/d"},
		{"///a/b//", "//c", "a/b/c"},
		{"/", "foo", "foo"},
		{"p", "foo/", "p/foo/"},
	}
	for _, c := range cases {
		got, err := joinKey(c.prefix, c.key)
		if err != nil || got != c.want {
			t.Errorf("joinKey(%q, %q) = %q, %v, want %q, nil", c.prefix, c.key, got, err, c.want)
		}
	}
}

func TestJoinKeyRejectsEmptyKey(t *testing.T) {
	for _, prefix := range []string{"", "p", "/p/"} {
		for _, key := range []string{"", "/", "//"} {
			got, err := joinKey(prefix, key)
			if !errors.Is(err, store.ErrInvalidKey) || got != "" {
				t.Errorf("joinKey(%q, %q) = %q, %v, want \"\", ErrInvalidKey", prefix, key, got, err)
			}
			if err != nil && !strings.HasPrefix(err.Error(), "azblob: ") {
				t.Errorf("joinKey(%q, %q) error %q lacks the backend name", prefix, key, err)
			}
		}
	}
}

// TestJoinKeyStaysInNamespace: every object a single-key operation addresses
// must lie inside the store's namespace, i.e. List must be able to map it
// back to the same key.
func TestJoinKeyStaysInNamespace(t *testing.T) {
	for _, prefix := range []string{"", "data", "/data/", "a/b"} {
		for _, key := range []string{"", "/", "//", "k", "/k", "k/v"} {
			obj, err := joinKey(prefix, key)
			if err != nil {
				continue // rejected keys address no object at all
			}
			got, ok := keyFromObject(prefix, obj)
			if want := strings.TrimLeft(key, "/"); !ok || got != want {
				t.Errorf("joinKey(%q, %q) = %q, outside the store namespace (keyFromObject = %q, %v)", prefix, key, obj, got, ok)
			}
		}
	}
}

func TestListPrefix(t *testing.T) {
	cases := []struct {
		storePrefix, prefix, want string
	}{
		{"", "", ""},
		{"", "foo", "foo"},
		{"", "/foo", "foo"},
		{"", "foo/", "foo/"},
		{"p", "", "p/"},
		{"p", "/", "p/"},
		{"p", "foo", "p/foo"},
		{"/p/", "/foo/", "p/foo/"},
		{"a/b", "c", "a/b/c"},
	}
	for _, c := range cases {
		got := listPrefix(c.storePrefix, c.prefix)
		if got != c.want {
			t.Errorf("listPrefix(%q, %q) = %q, want %q", c.storePrefix, c.prefix, got, c.want)
		}
	}
}

func TestKeyFromObject(t *testing.T) {
	cases := []struct {
		storePrefix, objKey, want string
		ok                        bool
	}{
		{"", "foo", "foo", true},
		{"", "", "", false},
		{"p", "p/foo", "foo", true},
		{"p", "p/foo/bar", "foo/bar", true},
		{"/p/", "p/foo", "foo", true},
		{"a/b", "a/b/c", "c", true},
		{"p", "p", "", false},
		{"p", "p/", "", false},
		{"p", "p2/foo", "", false},
		{"p", "other/foo", "", false},
	}
	for _, c := range cases {
		got, ok := keyFromObject(c.storePrefix, c.objKey)
		if got != c.want || ok != c.ok {
			t.Errorf("keyFromObject(%q, %q) = %q, %v, want %q, %v", c.storePrefix, c.objKey, got, ok, c.want, c.ok)
		}
	}
}

// listObjects simulates a backend List: the bucket returns every object whose
// name has the computed list prefix (string-prefix semantics), and each name
// is mapped back to a store key.
func listObjects(objects []string, storePrefix, prefix string) []string {
	full := listPrefix(storePrefix, prefix)
	var keys []string
	for _, obj := range objects {
		if !strings.HasPrefix(obj, full) {
			continue
		}
		if k, ok := keyFromObject(storePrefix, obj); ok {
			keys = append(keys, k)
		}
	}
	return keys
}

func TestListNamespace(t *testing.T) {
	objects := []string{"data", "data/", "data/a", "data/b/c", "data2/x", "database/y", "other"}
	cases := []struct {
		storePrefix, prefix string
		want                []string
	}{
		{"data", "", []string{"a", "b/c"}},
		{"data", "/", []string{"a", "b/c"}},
		{"data", "b", []string{"b/c"}},
		{"data", "b/", []string{"b/c"}},
		{"data", "a", []string{"a"}},
		{"data", "x", nil},
		{"/data/", "", []string{"a", "b/c"}},
		{"", "", objects},
		{"", "data", []string{"data", "data/", "data/a", "data/b/c", "data2/x", "database/y"}},
		{"", "data/", []string{"data/", "data/a", "data/b/c"}},
		{"", "/data/", []string{"data/", "data/a", "data/b/c"}},
	}
	for _, c := range cases {
		got := listObjects(objects, c.storePrefix, c.prefix)
		if !slices.Equal(got, c.want) {
			t.Errorf("List(storePrefix=%q, prefix=%q) = %q, want %q", c.storePrefix, c.prefix, got, c.want)
		}
	}
}

func TestClampLen(t *testing.T) {
	cases := []struct {
		off, n, want int64
	}{
		{0, 0, 0},
		{0, 10, 10},
		{5, 10, 10},
		{0, math.MaxInt64, math.MaxInt64},
		{5, math.MaxInt64, math.MaxInt64 - 5},
		{math.MaxInt64 - 1, 5, 1},
		{math.MaxInt64, 1, 0},
		{math.MaxInt64, math.MaxInt64, 0},
	}
	for _, c := range cases {
		got := clampLen(c.off, c.n)
		if got != c.want {
			t.Errorf("clampLen(%d, %d) = %d, want %d", c.off, c.n, got, c.want)
		}
		if end := c.off + got; end < c.off {
			t.Errorf("clampLen(%d, %d) = %d: off+n overflows to %d", c.off, c.n, got, end)
		}
	}
}

func TestCheckRange(t *testing.T) {
	if err := checkRange(0, 0); err != nil {
		t.Errorf("checkRange(0, 0) = %v, want nil", err)
	}
	if err := checkRange(-1, 0); !errors.Is(err, store.ErrInvalidRange) {
		t.Errorf("checkRange(-1, 0) = %v, want ErrInvalidRange", err)
	}
	if err := checkRange(0, -1); !errors.Is(err, store.ErrInvalidRange) {
		t.Errorf("checkRange(0, -1) = %v, want ErrInvalidRange", err)
	}
	if err := checkSuffix(0); err != nil {
		t.Errorf("checkSuffix(0) = %v, want nil", err)
	}
	if err := checkSuffix(-1); !errors.Is(err, store.ErrInvalidRange) {
		t.Errorf("checkSuffix(-1) = %v, want ErrInvalidRange", err)
	}
}
