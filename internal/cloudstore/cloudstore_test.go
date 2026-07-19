package cloudstore

import (
	"errors"
	"testing"

	"github.com/rkm/zarr-go/store"
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
		{"p", "", "p"},
		{"", "", ""},
		{"p", "/", "p"},
	}
	for _, c := range cases {
		got := JoinKey(c.prefix, c.key)
		if got != c.want {
			t.Errorf("JoinKey(%q, %q) = %q, want %q", c.prefix, c.key, got, c.want)
		}
	}
}

func TestStripPrefix(t *testing.T) {
	cases := []struct {
		prefix, key, want string
	}{
		{"", "foo", "foo"},
		{"p", "p/foo", "foo"},
		{"p", "p/foo/bar", "foo/bar"},
		{"a/b", "a/b/c", "c"},
		// key without prefix is returned as-is.
		{"p", "other/foo", "other/foo"},
	}
	for _, c := range cases {
		got := StripPrefix(c.prefix, c.key)
		if got != c.want {
			t.Errorf("StripPrefix(%q, %q) = %q, want %q", c.prefix, c.key, got, c.want)
		}
	}
}

func TestTrimSlashes(t *testing.T) {
	cases := map[string]string{
		"":         "",
		"/":        "",
		"//a//":    "a",
		"a/b":      "a/b",
		"/a/b/":    "a/b",
		"///a/b//": "a/b",
	}
	for in, want := range cases {
		if got := TrimSlashes(in); got != want {
			t.Errorf("TrimSlashes(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestValidateRange(t *testing.T) {
	if err := ValidateRange("s3", "GetRange", 0, 0); err != nil {
		t.Errorf("ValidateRange(0,0) = %v, want nil", err)
	}
	if err := ValidateRange("s3", "GetRange", -1, 0); !errors.Is(err, store.ErrInvalidRange) {
		t.Errorf("ValidateRange(-1,0) = %v, want ErrInvalidRange", err)
	}
	if err := ValidateRange("s3", "GetRange", 0, -1); !errors.Is(err, store.ErrInvalidRange) {
		t.Errorf("ValidateRange(0,-1) = %v, want ErrInvalidRange", err)
	}
	if err := ValidateRange("s3", "GetSuffix", 0, -1); !errors.Is(err, store.ErrInvalidRange) {
		t.Errorf("ValidateRange GetSuffix(-1) = %v, want ErrInvalidRange", err)
	}
}
