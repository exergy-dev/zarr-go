package fs

import (
	"context"
	"errors"
	"sort"
	"testing"

	"github.com/rkm/zarr-go/store"
)

func newStore(t *testing.T) *Store {
	t.Helper()
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

func TestNewRequiresExistingDirectory(t *testing.T) {
	if _, err := New("/path/that/does/not/exist/zzz"); err == nil {
		t.Fatal("expected error for missing root, got nil")
	}
}

func TestSetGetRoundTrip(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)

	want := []byte("hello, zarr")
	if err := s.Set(ctx, "a/b/c.bin", want); err != nil {
		t.Fatalf("Set: %v", err)
	}
	got, err := s.Get(ctx, "a/b/c.bin")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("Get: got %q, want %q", got, want)
	}

	ok, err := s.Exists(ctx, "a/b/c.bin")
	if err != nil || !ok {
		t.Fatalf("Exists: ok=%v err=%v", ok, err)
	}
}

func TestGetRangeSubSlice(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)

	full := []byte("0123456789ABCDEF")
	if err := s.Set(ctx, "data", full); err != nil {
		t.Fatalf("Set: %v", err)
	}

	got, err := s.GetRange(ctx, "data", 4, 6)
	if err != nil {
		t.Fatalf("GetRange: %v", err)
	}
	if string(got) != "456789" {
		t.Fatalf("GetRange: got %q, want %q", got, "456789")
	}

	// Clamping: ask for more than file size.
	got, err = s.GetRange(ctx, "data", 10, 100)
	if err != nil {
		t.Fatalf("GetRange clamp: %v", err)
	}
	if string(got) != "ABCDEF" {
		t.Fatalf("GetRange clamp: got %q, want %q", got, "ABCDEF")
	}

	// Offset past EOF: empty.
	got, err = s.GetRange(ctx, "data", 999, 4)
	if err != nil {
		t.Fatalf("GetRange past eof: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("GetRange past eof: got %q, want empty", got)
	}
}

func TestListRecursive(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)

	keys := []string{
		"arr/zarr.json",
		"arr/c/0/0",
		"arr/c/0/1",
		"arr/c/1/0",
		"other/zarr.json",
	}
	for _, k := range keys {
		if err := s.Set(ctx, k, []byte("x")); err != nil {
			t.Fatalf("Set %q: %v", k, err)
		}
	}

	var got []string
	for k, err := range s.List(ctx, "arr") {
		if err != nil {
			t.Fatalf("List: %v", err)
		}
		got = append(got, k)
	}
	sort.Strings(got)

	want := []string{"arr/c/0/0", "arr/c/0/1", "arr/c/1/0", "arr/zarr.json"}
	if len(got) != len(want) {
		t.Fatalf("List length: got %v, want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("List[%d]: got %q, want %q", i, got[i], want[i])
		}
	}

	// Missing prefix: no error, no items.
	count := 0
	for _, err := range s.List(ctx, "no/such/prefix") {
		if err != nil {
			t.Fatalf("List missing: %v", err)
		}
		count++
	}
	if count != 0 {
		t.Fatalf("List missing: got %d items, want 0", count)
	}
}

func TestGetMissingReturnsErrKeyNotFound(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)

	_, err := s.Get(ctx, "missing/key")
	if err == nil {
		t.Fatal("Get missing: expected error, got nil")
	}
	if !errors.Is(err, store.ErrKeyNotFound) {
		t.Fatalf("Get missing: err = %v, want store.ErrKeyNotFound", err)
	}

	// GetRange on missing key as well.
	_, err = s.GetRange(ctx, "missing/key", 0, 4)
	if !errors.Is(err, store.ErrKeyNotFound) {
		t.Fatalf("GetRange missing: err = %v, want store.ErrKeyNotFound", err)
	}
}

func TestSetIfNotExists(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)

	ok, err := s.SetIfNotExists(ctx, "lock", []byte("first"))
	if err != nil || !ok {
		t.Fatalf("first SetIfNotExists: ok=%v err=%v", ok, err)
	}

	ok, err = s.SetIfNotExists(ctx, "lock", []byte("second"))
	if err != nil {
		t.Fatalf("second SetIfNotExists: err=%v", err)
	}
	if ok {
		t.Fatal("second SetIfNotExists: ok=true, want false")
	}

	got, err := s.Get(ctx, "lock")
	if err != nil {
		t.Fatalf("Get after CAS: %v", err)
	}
	if string(got) != "first" {
		t.Fatalf("Get after CAS: got %q, want %q", got, "first")
	}
}

func TestRejectDotDotInKey(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)

	for _, k := range []string{"../escape", "a/../b", "..", "a/b/.."} {
		if err := s.Set(ctx, k, []byte("x")); err == nil {
			t.Errorf("Set %q: expected error, got nil", k)
		}
		if _, err := s.Get(ctx, k); err == nil {
			t.Errorf("Get %q: expected error, got nil", k)
		}
		if _, err := s.Exists(ctx, k); err == nil {
			t.Errorf("Exists %q: expected error, got nil", k)
		}
	}
}

func TestDeleteMissingNoError(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)

	if err := s.Delete(ctx, "never/existed"); err != nil {
		t.Fatalf("Delete missing: %v", err)
	}

	if err := s.Set(ctx, "k", []byte("v")); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if err := s.Delete(ctx, "k"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	ok, err := s.Exists(ctx, "k")
	if err != nil {
		t.Fatalf("Exists: %v", err)
	}
	if ok {
		t.Fatal("Exists after Delete: got true, want false")
	}
}
