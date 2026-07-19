// Package storetest is a shared conformance test suite for store.Store
// implementations. Every backend's own test file should call RunConformance
// against a fresh instance so the store.Store contract (documented in
// store/store.go) is enforced identically across all of them — the
// divergences this suite exists to catch (inconsistent GetRange/List
// semantics between backends) are exactly the kind of bug that a per-backend
// test suite, written independently, tends to miss.
package storetest

import (
	"context"
	"errors"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/rkm/zarr-go/store"
)

// Options configures RunConformance for backends with non-standard write
// semantics.
type Options struct {
	// Seed writes a key/value pair into the store out-of-band. If nil,
	// RunConformance seeds via s.(store.WritableStore).Set — which requires
	// the store to actually persist writes (not a read-only stub).
	Seed func(t *testing.T, s store.Store, key string, value []byte)

	// ReadOnly marks the backend as not supporting Set/Delete/SetIfNotExists
	// (e.g. the http store). Mutation-focused subtests are skipped; Seed
	// must be non-nil when ReadOnly is true.
	ReadOnly bool

	// NoList marks the backend as not supporting List at all (e.g. the http
	// store, which has no portable directory-listing protocol and always
	// returns ErrListUnsupported). The List subtest is skipped.
	NoList bool
}

// RunConformance runs the shared store.Store contract test matrix. newStore
// is called once per subtest (and once per table-driven case within some
// subtests) to obtain a store instance; backends that share persistent state
// across "fresh" instances (e.g. a real bucket reused between calls) must
// use collision-free key names, which RunConformance ensures by namespacing
// every key it writes under t.Name(). Every seeded key is deleted via
// t.Cleanup after use (best-effort, ignoring errors — some ReadOnly
// backends can't delete), so repeated runs against a persistent real
// backend (e.g. a shared test bucket) don't collide with leftover state
// from a prior run.
func RunConformance(t *testing.T, newStore func(t *testing.T) store.Store, opts Options) {
	t.Helper()
	baseSeed := opts.Seed
	if baseSeed == nil {
		baseSeed = func(t *testing.T, s store.Store, key string, value []byte) {
			t.Helper()
			if err := s.Set(context.Background(), key, value); err != nil {
				t.Fatalf("seed Set(%q): %v", key, err)
			}
		}
	}
	seed := func(t *testing.T, s store.Store, key string, value []byte) {
		t.Helper()
		baseSeed(t, s, key, value)
		if !opts.ReadOnly {
			t.Cleanup(func() { _ = s.Delete(context.Background(), key) })
		}
	}

	t.Run("Basics", func(t *testing.T) { testBasics(t, newStore, seed) })
	t.Run("GetRange", func(t *testing.T) { testGetRange(t, newStore, seed) })
	t.Run("GetSuffix", func(t *testing.T) { testGetSuffix(t, newStore, seed) })
	if !opts.NoList {
		t.Run("List", func(t *testing.T) { testList(t, newStore, seed) })
	}
	if !opts.ReadOnly {
		t.Run("SetIfNotExists", func(t *testing.T) { testSetIfNotExists(t, newStore) })
		t.Run("DeleteMissing", func(t *testing.T) { testDeleteMissing(t, newStore) })
	}
}

func testBasics(t *testing.T, newStore func(t *testing.T) store.Store, seed func(*testing.T, store.Store, string, []byte)) {
	s := newStore(t)
	ctx := context.Background()
	key := t.Name() + "/key"

	if _, err := s.Get(ctx, key); !errors.Is(err, store.ErrKeyNotFound) {
		t.Fatalf("Get(missing) = %v, want ErrKeyNotFound", err)
	}
	if !errors.Is(store.ErrKeyNotFound, os.ErrNotExist) {
		t.Fatal("store.ErrKeyNotFound must satisfy errors.Is(_, os.ErrNotExist)")
	}
	if ok, err := s.Exists(ctx, key); err != nil || ok {
		t.Fatalf("Exists(missing) = %v,%v want false,nil", ok, err)
	}

	val := []byte("hello world")
	seed(t, s, key, val)

	got, err := s.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if string(got) != string(val) {
		t.Fatalf("Get = %q, want %q", got, val)
	}
	if ok, err := s.Exists(ctx, key); err != nil || !ok {
		t.Fatalf("Exists = %v,%v want true,nil", ok, err)
	}
}

func testDeleteMissing(t *testing.T, newStore func(t *testing.T) store.Store) {
	s := newStore(t)
	ctx := context.Background()
	if err := s.Delete(ctx, t.Name()+"/does/not/exist"); err != nil {
		t.Fatalf("Delete(missing) = %v, want nil", err)
	}
}

func testGetRange(t *testing.T, newStore func(t *testing.T) store.Store, seed func(*testing.T, store.Store, string, []byte)) {
	s := newStore(t)
	ctx := context.Background()
	key := t.Name() + "/key"
	val := []byte("0123456789")
	seed(t, s, key, val)
	missing := t.Name() + "/missing"

	t.Run("negative_off", func(t *testing.T) {
		if _, err := s.GetRange(ctx, key, -1, 4); !errors.Is(err, store.ErrInvalidRange) {
			t.Fatalf("GetRange(off=-1) = %v, want ErrInvalidRange", err)
		}
	})
	t.Run("negative_n", func(t *testing.T) {
		if _, err := s.GetRange(ctx, key, 0, -1); !errors.Is(err, store.ErrInvalidRange) {
			t.Fatalf("GetRange(n=-1) = %v, want ErrInvalidRange", err)
		}
	})
	t.Run("n=0_existing_key", func(t *testing.T) {
		got, err := s.GetRange(ctx, key, 3, 0)
		if err != nil {
			t.Fatalf("GetRange(n=0): %v", err)
		}
		if len(got) != 0 {
			t.Fatalf("GetRange(n=0) = %q, want empty", got)
		}
	})
	t.Run("n=0_missing_key", func(t *testing.T) {
		if _, err := s.GetRange(ctx, missing, 0, 0); !errors.Is(err, store.ErrKeyNotFound) {
			t.Fatalf("GetRange(missing,n=0) = %v, want ErrKeyNotFound", err)
		}
	})
	t.Run("off_past_EOF", func(t *testing.T) {
		got, err := s.GetRange(ctx, key, int64(len(val))+5, 3)
		if err != nil {
			t.Fatalf("GetRange(off past EOF): %v", err)
		}
		if len(got) != 0 {
			t.Fatalf("GetRange(off past EOF) = %q, want empty", got)
		}
	})
	t.Run("off_at_EOF", func(t *testing.T) {
		got, err := s.GetRange(ctx, key, int64(len(val)), 3)
		if err != nil {
			t.Fatalf("GetRange(off==size): %v", err)
		}
		if len(got) != 0 {
			t.Fatalf("GetRange(off==size) = %q, want empty", got)
		}
	})
	t.Run("clamp_past_EOF", func(t *testing.T) {
		got, err := s.GetRange(ctx, key, 5, 100)
		if err != nil {
			t.Fatalf("GetRange(clamp): %v", err)
		}
		if string(got) != string(val[5:]) {
			t.Fatalf("GetRange(clamp) = %q, want %q", got, val[5:])
		}
	})
	t.Run("exact_range", func(t *testing.T) {
		got, err := s.GetRange(ctx, key, 2, 4)
		if err != nil {
			t.Fatalf("GetRange: %v", err)
		}
		if string(got) != string(val[2:6]) {
			t.Fatalf("GetRange = %q, want %q", got, val[2:6])
		}
	})
	t.Run("missing_key", func(t *testing.T) {
		if _, err := s.GetRange(ctx, missing, 0, 4); !errors.Is(err, store.ErrKeyNotFound) {
			t.Fatalf("GetRange(missing) = %v, want ErrKeyNotFound", err)
		}
	})
}

func testGetSuffix(t *testing.T, newStore func(t *testing.T) store.Store, seed func(*testing.T, store.Store, string, []byte)) {
	s := newStore(t)
	sr, ok := s.(store.SuffixReader)
	if !ok {
		t.Skip("store does not implement store.SuffixReader")
	}
	ctx := context.Background()
	key := t.Name() + "/key"
	val := []byte("0123456789")
	seed(t, s, key, val)
	missing := t.Name() + "/missing"

	t.Run("negative_n", func(t *testing.T) {
		if _, err := sr.GetSuffix(ctx, key, -1); !errors.Is(err, store.ErrInvalidRange) {
			t.Fatalf("GetSuffix(n=-1) = %v, want ErrInvalidRange", err)
		}
	})
	t.Run("n=0_existing_key", func(t *testing.T) {
		got, err := sr.GetSuffix(ctx, key, 0)
		if err != nil {
			t.Fatalf("GetSuffix(n=0): %v", err)
		}
		if len(got) != 0 {
			t.Fatalf("GetSuffix(n=0) = %q, want empty", got)
		}
	})
	t.Run("n=0_missing_key", func(t *testing.T) {
		if _, err := sr.GetSuffix(ctx, missing, 0); !errors.Is(err, store.ErrKeyNotFound) {
			t.Fatalf("GetSuffix(missing,n=0) = %v, want ErrKeyNotFound", err)
		}
	})
	t.Run("interior", func(t *testing.T) {
		got, err := sr.GetSuffix(ctx, key, 4)
		if err != nil {
			t.Fatalf("GetSuffix(4): %v", err)
		}
		want := val[len(val)-4:]
		if string(got) != string(want) {
			t.Fatalf("GetSuffix(4) = %q, want %q", got, want)
		}
	})
	t.Run("n=size", func(t *testing.T) {
		got, err := sr.GetSuffix(ctx, key, int64(len(val)))
		if err != nil {
			t.Fatalf("GetSuffix(size): %v", err)
		}
		if string(got) != string(val) {
			t.Fatalf("GetSuffix(size) = %q, want %q", got, val)
		}
	})
	t.Run("n>size", func(t *testing.T) {
		got, err := sr.GetSuffix(ctx, key, int64(len(val))+100)
		if err != nil {
			t.Fatalf("GetSuffix(n>size): %v", err)
		}
		if string(got) != string(val) {
			t.Fatalf("GetSuffix(n>size) = %q, want %q", got, val)
		}
	})
	t.Run("missing_key", func(t *testing.T) {
		if _, err := sr.GetSuffix(ctx, missing, 4); !errors.Is(err, store.ErrKeyNotFound) {
			t.Fatalf("GetSuffix(missing) = %v, want ErrKeyNotFound", err)
		}
	})
}

func testList(t *testing.T, newStore func(t *testing.T) store.Store, seed func(*testing.T, store.Store, string, []byte)) {
	s := newStore(t)
	ctx := context.Background()
	root := t.Name() + "/"
	// Deliberately avoid a key that is both a leaf (e.g. "dir") and, via a
	// slash, a directory prefix of another key (e.g. "dir/a") — flat
	// object-store backends (mem/s3/gcs/azblob) support that fine, but a
	// filesystem-backed store cannot represent both a file and a directory
	// at the same path simultaneously. "dirx" tests the pure-string-prefix
	// case (it shares the literal prefix "dir" without being nested under
	// it) without creating that collision.
	suffixes := []string{"x", "xy", "dir/a", "dir/ab", "dirx"}
	for _, suf := range suffixes {
		seed(t, s, root+suf, []byte("v"))
	}

	cases := []struct {
		name   string
		prefix string
		want   []string
	}{
		{"sibling_prefix", root + "x", []string{root + "x", root + "xy"}},
		// The original bug this guards: a prefix that is a partial path
		// segment (not ending in "/") must still match via plain string
		// prefix, not just exact-directory matching.
		{"partial_segment_after_slash", root + "dir/a", []string{root + "dir/a", root + "dir/ab"}},
		{"segment_with_slash", root + "dir/", []string{root + "dir/a", root + "dir/ab"}},
		{"prefix_spans_leaf_and_dir", root + "dir", []string{root + "dir/a", root + "dir/ab", root + "dirx"}},
		{"whole_root", root, []string{root + "x", root + "xy", root + "dir/a", root + "dir/ab", root + "dirx"}},
		{"no_match", root + "nomatch", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got []string
			for k, err := range s.List(ctx, tc.prefix) {
				if err != nil {
					t.Fatalf("List(%q): %v", tc.prefix, err)
				}
				if !strings.HasPrefix(k, tc.prefix) {
					t.Fatalf("List(%q) yielded %q, which doesn't have that prefix", tc.prefix, k)
				}
				got = append(got, k)
			}
			sort.Strings(got)
			want := append([]string(nil), tc.want...)
			sort.Strings(want)
			if len(got) != len(want) {
				t.Fatalf("List(%q) = %v, want %v", tc.prefix, got, want)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("List(%q)[%d] = %q, want %q (full: got=%v want=%v)", tc.prefix, i, got[i], want[i], got, want)
				}
			}
		})
	}
}

func testSetIfNotExists(t *testing.T, newStore func(t *testing.T) store.Store) {
	s := newStore(t)
	ws, ok := s.(store.WritableStore)
	if !ok {
		t.Skip("store does not implement store.WritableStore")
	}
	ctx := context.Background()
	key := t.Name() + "/key"
	t.Cleanup(func() { _ = s.Delete(context.Background(), key) })

	ok1, err := ws.SetIfNotExists(ctx, key, []byte("first"))
	if err != nil || !ok1 {
		t.Fatalf("first SetIfNotExists: ok=%v err=%v, want true,nil", ok1, err)
	}
	ok2, err := ws.SetIfNotExists(ctx, key, []byte("second"))
	if err != nil || ok2 {
		t.Fatalf("second SetIfNotExists: ok=%v err=%v, want false,nil", ok2, err)
	}
	got, err := s.Get(ctx, key)
	if err != nil || string(got) != "first" {
		t.Fatalf("Get after race: got %q err=%v, want %q,nil", got, err, "first")
	}
}
