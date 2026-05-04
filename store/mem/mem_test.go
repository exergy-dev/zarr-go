package mem_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/rkm/zarr-go/store"
	"github.com/rkm/zarr-go/store/mem"
)

func TestSetGetRoundTripAndCopySemantics(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := mem.New()

	in := []byte("hello world")
	if err := s.Set(ctx, "k", in); err != nil {
		t.Fatalf("Set: %v", err)
	}

	// Mutating the input after Set must not affect the store.
	in[0] = 'X'

	got, err := s.Get(ctx, "k")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if string(got) != "hello world" {
		t.Fatalf("Set did not defensively copy: got %q", got)
	}

	// Mutating the returned slice must not affect the store either.
	got[0] = 'Y'
	got2, err := s.Get(ctx, "k")
	if err != nil {
		t.Fatalf("Get (second): %v", err)
	}
	if string(got2) != "hello world" {
		t.Fatalf("Get did not defensively copy: got %q", got2)
	}
}

func TestGetRange(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := mem.New()
	if err := s.Set(ctx, "k", []byte("0123456789")); err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		off, n int64
		want   string
	}{
		{0, 4, "0123"},
		{2, 3, "234"},
		{5, 100, "56789"}, // clamped
		{10, 5, ""},       // off == len
	}
	for _, c := range cases {
		got, err := s.GetRange(ctx, "k", c.off, c.n)
		if err != nil {
			t.Fatalf("GetRange(%d,%d): %v", c.off, c.n, err)
		}
		if string(got) != c.want {
			t.Fatalf("GetRange(%d,%d) = %q, want %q", c.off, c.n, got, c.want)
		}
	}

	// Mutating returned range bytes must not affect the store.
	r, err := s.GetRange(ctx, "k", 0, 4)
	if err != nil {
		t.Fatal(err)
	}
	r[0] = 'X'
	full, err := s.Get(ctx, "k")
	if err != nil {
		t.Fatal(err)
	}
	if string(full) != "0123456789" {
		t.Fatalf("GetRange leaked storage: %q", full)
	}

	// Negative off.
	if _, err := s.GetRange(ctx, "k", -1, 1); err == nil {
		t.Fatal("expected error for negative offset")
	}
	// off > length: empty slice, no error (matches fs/s3/gcs/azblob).
	past, err := s.GetRange(ctx, "k", 11, 1)
	if err != nil {
		t.Fatalf("GetRange past EOF: %v", err)
	}
	if len(past) != 0 {
		t.Fatalf("GetRange past EOF = %q, want empty", past)
	}
	// Missing key.
	if _, err := s.GetRange(ctx, "missing", 0, 1); !errors.Is(err, store.ErrKeyNotFound) {
		t.Fatalf("missing key: got %v, want ErrKeyNotFound", err)
	}
}

func TestMissingKeyErrors(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := mem.New()

	_, err := s.Get(ctx, "nope")
	if !errors.Is(err, store.ErrKeyNotFound) {
		t.Fatalf("Get missing: got %v, want ErrKeyNotFound", err)
	}
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("Get missing: err should wrap os.ErrNotExist; got %v", err)
	}

	ok, err := s.Exists(ctx, "nope")
	if err != nil {
		t.Fatalf("Exists: %v", err)
	}
	if ok {
		t.Fatal("Exists returned true for missing key")
	}

	// Delete on missing key must succeed silently.
	if err := s.Delete(ctx, "nope"); err != nil {
		t.Fatalf("Delete missing: %v", err)
	}
}

func TestListPrefixAndOrder(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := mem.New()

	keys := []string{"b/2", "a/3", "a/1", "a/2", "c/1"}
	for _, k := range keys {
		if err := s.Set(ctx, k, []byte(k)); err != nil {
			t.Fatal(err)
		}
	}

	var got []string
	for k, err := range s.List(ctx, "a/") {
		if err != nil {
			t.Fatalf("List error: %v", err)
		}
		got = append(got, k)
	}
	want := []string{"a/1", "a/2", "a/3"}
	if len(got) != len(want) {
		t.Fatalf("List len: got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("List[%d]: got %q, want %q (full=%v)", i, got[i], want[i], got)
		}
	}

	// Empty prefix yields everything in sorted order.
	var all []string
	for k, err := range s.List(ctx, "") {
		if err != nil {
			t.Fatal(err)
		}
		all = append(all, k)
	}
	wantAll := []string{"a/1", "a/2", "a/3", "b/2", "c/1"}
	if fmt.Sprint(all) != fmt.Sprint(wantAll) {
		t.Fatalf("List(\"\"): got %v, want %v", all, wantAll)
	}

	// Prefix with no matches yields nothing.
	for k, err := range s.List(ctx, "z/") {
		t.Fatalf("List(z/) yielded %q (err=%v); expected no results", k, err)
	}
}

func TestSetIfNotExists(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := mem.New()

	ok, err := s.SetIfNotExists(ctx, "k", []byte("first"))
	if err != nil {
		t.Fatalf("SetIfNotExists first: %v", err)
	}
	if !ok {
		t.Fatal("first SetIfNotExists returned false")
	}

	ok, err = s.SetIfNotExists(ctx, "k", []byte("second"))
	if err != nil {
		t.Fatalf("SetIfNotExists second: %v", err)
	}
	if ok {
		t.Fatal("second SetIfNotExists returned true")
	}

	got, err := s.Get(ctx, "k")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "first" {
		t.Fatalf("value clobbered: %q", got)
	}
}

func TestConcurrentAccess(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	s := mem.New()

	const writers = 8
	const readers = 8
	const perGoroutine = 200

	var wg sync.WaitGroup
	wg.Add(writers + readers)

	for w := 0; w < writers; w++ {
		go func(id int) {
			defer wg.Done()
			for i := 0; i < perGoroutine; i++ {
				key := fmt.Sprintf("w%d/%d", id, i)
				val := []byte(key)
				if err := s.Set(ctx, key, val); err != nil {
					t.Errorf("Set: %v", err)
					return
				}
				if i%5 == 0 {
					if _, err := s.SetIfNotExists(ctx, "shared/"+key, val); err != nil {
						t.Errorf("SetIfNotExists: %v", err)
						return
					}
				}
				if i%7 == 0 {
					if err := s.Delete(ctx, key); err != nil {
						t.Errorf("Delete: %v", err)
						return
					}
				}
			}
		}(w)
	}

	for r := 0; r < readers; r++ {
		go func(id int) {
			defer wg.Done()
			for i := 0; i < perGoroutine; i++ {
				key := fmt.Sprintf("w%d/%d", id%writers, i)
				if _, err := s.Get(ctx, key); err != nil && !errors.Is(err, store.ErrKeyNotFound) {
					t.Errorf("Get: %v", err)
					return
				}
				if _, err := s.Exists(ctx, key); err != nil {
					t.Errorf("Exists: %v", err)
					return
				}
				count := 0
				for _, err := range s.List(ctx, "w") {
					if err != nil {
						t.Errorf("List: %v", err)
						return
					}
					count++
					if count > 50 { // bound work
						break
					}
				}
			}
		}(r)
	}

	wg.Wait()
}

func TestContextCancellation(t *testing.T) {
	t.Parallel()
	s := mem.New()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := s.Get(ctx, "k"); err == nil {
		t.Fatal("Get: expected error from canceled context")
	}
	if err := s.Set(ctx, "k", []byte("v")); err == nil {
		t.Fatal("Set: expected error from canceled context")
	}
}
