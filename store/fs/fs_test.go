package fs

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/exergy-dev/zarr-go/store"
	"github.com/exergy-dev/zarr-go/store/storetest"
)

func newStore(t *testing.T) *Store {
	t.Helper()
	s, err := New(t.TempDir())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

func TestConformance(t *testing.T) {
	storetest.RunConformance(t, func(t *testing.T) store.Store {
		return newStore(t)
	}, storetest.Options{})
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

// TestSetDoesNotClobberSiblingTmpKey checks that Set("a") does not disturb
// an unrelated, legitimately-named key "a.tmp".
func TestSetDoesNotClobberSiblingTmpKey(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)

	if err := s.Set(ctx, "a.tmp", []byte("sibling")); err != nil {
		t.Fatalf("Set a.tmp: %v", err)
	}
	if err := s.Set(ctx, "a", []byte("main")); err != nil {
		t.Fatalf("Set a: %v", err)
	}
	got, err := s.Get(ctx, "a.tmp")
	if err != nil {
		t.Fatalf("Get a.tmp: %v", err)
	}
	if string(got) != "sibling" {
		t.Fatalf("Get a.tmp = %q, want %q (Set(\"a\") clobbered a sibling key)", got, "sibling")
	}
}

// TestConcurrentSetSameKeyNeverTorn checks that concurrent Set calls to the
// same key never produce a torn value: each writer stages in its own temp
// file, so every observed value must be byte-identical to exactly one
// writer's full payload.
func TestConcurrentSetSameKeyNeverTorn(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)

	const writers = 8
	const size = 64 * 1024
	payloads := make([][]byte, writers)
	for i := range payloads {
		payloads[i] = make([]byte, size)
		for j := range payloads[i] {
			payloads[i][j] = byte(i)
		}
	}

	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			if err := s.Set(ctx, "shared", payloads[i]); err != nil {
				t.Errorf("Set (writer %d): %v", i, err)
			}
		}(i)
	}
	close(start)
	wg.Wait()

	got, err := s.Get(ctx, "shared")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	matched := -1
	for i, p := range payloads {
		if bytes.Equal(got, p) {
			matched = i
			break
		}
	}
	if matched == -1 {
		t.Fatalf("final value (len %d) does not match any single writer's payload — torn write", len(got))
	}
}

// TestListFiltersStagingFiles guards against a crash-leaked temp file (or
// one observed mid-write) ever surfacing as a phantom store key.
func TestListFiltersStagingFiles(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)

	if err := s.Set(ctx, "real", []byte("v")); err != nil {
		t.Fatalf("Set: %v", err)
	}
	path, err := s.resolve("real")
	if err != nil {
		t.Fatal(err)
	}
	stray := filepath.Join(filepath.Dir(path), tempPrefix+"leaked123")
	if err := os.WriteFile(stray, []byte("leaked"), 0o644); err != nil {
		t.Fatalf("plant stray temp file: %v", err)
	}

	var got []string
	for k, err := range s.List(ctx, "") {
		if err != nil {
			t.Fatalf("List: %v", err)
		}
		got = append(got, k)
	}
	if len(got) != 1 || got[0] != "real" {
		t.Fatalf("List = %v, want [\"real\"] (staging file leaked into results)", got)
	}
}

// TestResolveRejectsReservedTempSegment checks that a user-supplied key
// colliding with the reserved staging namespace is rejected outright,
// rather than being silently treated as an ordinary key that List then
// filters away.
func TestResolveRejectsReservedTempSegment(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)
	if err := s.Set(ctx, tempPrefix+"whatever", []byte("x")); !errors.Is(err, store.ErrInvalidKey) {
		t.Fatalf("Set(reserved) = %v, want ErrInvalidKey", err)
	}
}

// TestSetIfNotExistsNoTornVisibility uses testHookBeforeCommit to pause
// SetIfNotExists after the temp file is fully written but before the
// publishing hard-link, and checks that a concurrent Exists/Get sees no
// trace of the key (no zero-byte or partially-written file).
func TestSetIfNotExistsNoTornVisibility(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)

	release := make(chan struct{})
	reached := make(chan struct{})
	testHookBeforeCommit = func() {
		close(reached)
		<-release
	}
	t.Cleanup(func() { testHookBeforeCommit = nil })

	done := make(chan struct{})
	go func() {
		defer close(done)
		ok, err := s.SetIfNotExists(ctx, "pending", []byte("full-value"))
		if err != nil || !ok {
			t.Errorf("SetIfNotExists: ok=%v err=%v", ok, err)
		}
	}()

	<-reached
	if ok, err := s.Exists(ctx, "pending"); err != nil || ok {
		t.Fatalf("Exists mid-write: ok=%v err=%v, want false,nil (key must not be visible before commit)", ok, err)
	}
	close(release)
	<-done

	got, err := s.Get(ctx, "pending")
	if err != nil || string(got) != "full-value" {
		t.Fatalf("Get after commit: got %q err=%v, want %q,nil", got, err, "full-value")
	}
}

// TestSetIfNotExistsConcurrentRaceExactlyOneWinner runs many concurrent
// SetIfNotExists calls for the same key with distinct large payloads and
// checks exactly one wins and the final value is byte-identical to that
// winner's payload (no interleaving between the loser's temp write and the
// winner's link).
func TestSetIfNotExistsConcurrentRaceExactlyOneWinner(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)

	const n = 16
	const size = 64 * 1024
	payloads := make([][]byte, n)
	for i := range payloads {
		payloads[i] = make([]byte, size)
		for j := range payloads[i] {
			payloads[i][j] = byte(i + 1)
		}
	}

	var wins int32
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			ok, err := s.SetIfNotExists(ctx, "race", payloads[i])
			if err != nil {
				t.Errorf("SetIfNotExists (%d): %v", i, err)
				return
			}
			if ok {
				atomic.AddInt32(&wins, 1)
			}
		}(i)
	}
	close(start)
	wg.Wait()

	if wins != 1 {
		t.Fatalf("wins = %d, want exactly 1", wins)
	}
	got, err := s.Get(ctx, "race")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	matched := false
	for _, p := range payloads {
		if bytes.Equal(got, p) {
			matched = true
			break
		}
	}
	if !matched {
		t.Fatalf("final value (len %d) does not match any candidate payload — torn write", len(got))
	}
}

func TestRejectDotDotInKey(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)

	for _, k := range []string{"../escape", "a/../b", "..", "a/b/.."} {
		if err := s.Set(ctx, k, []byte("x")); !errors.Is(err, store.ErrInvalidKey) {
			t.Errorf("Set %q: got %v, want ErrInvalidKey", k, err)
		}
		if _, err := s.Get(ctx, k); !errors.Is(err, store.ErrInvalidKey) {
			t.Errorf("Get %q: got %v, want ErrInvalidKey", k, err)
		}
		if _, err := s.Exists(ctx, k); !errors.Is(err, store.ErrInvalidKey) {
			t.Errorf("Exists %q: got %v, want ErrInvalidKey", k, err)
		}
	}
}

// TestResolveRejectsEscapingKeys checks that no key can address a path
// outside the root, including OS-separator spellings of ".." that a split
// on "/" alone misses (on windows, `..\x` is a single "/"-segment but
// filepath.Join treats it as a parent reference).
func TestResolveRejectsEscapingKeys(t *testing.T) {
	ctx := context.Background()
	parent := t.TempDir()
	root := filepath.Join(parent, "root")
	if err := os.Mkdir(root, 0o755); err != nil {
		t.Fatal(err)
	}
	s, err := New(root)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	keys := []string{"../x", "a/../../x", "..", "/../x"}
	if runtime.GOOS == "windows" {
		keys = append(keys, `..\x`, `a\..\..\x`, `C:x`, `C:\x`, `\x`)
	}
	for _, k := range keys {
		if err := s.Set(ctx, k, []byte("x")); !errors.Is(err, store.ErrInvalidKey) {
			t.Errorf("Set %q: got %v, want ErrInvalidKey", k, err)
		}
		if ok, err := s.SetIfNotExists(ctx, k, []byte("x")); !errors.Is(err, store.ErrInvalidKey) {
			t.Errorf("SetIfNotExists %q: ok=%v err=%v, want ErrInvalidKey", k, ok, err)
		}
		if _, err := s.Get(ctx, k); !errors.Is(err, store.ErrInvalidKey) {
			t.Errorf("Get %q: got %v, want ErrInvalidKey", k, err)
		}
		if _, err := s.GetRange(ctx, k, 0, 1); !errors.Is(err, store.ErrInvalidKey) {
			t.Errorf("GetRange %q: got %v, want ErrInvalidKey", k, err)
		}
		if _, err := s.GetSuffix(ctx, k, 1); !errors.Is(err, store.ErrInvalidKey) {
			t.Errorf("GetSuffix %q: got %v, want ErrInvalidKey", k, err)
		}
		if _, err := s.Exists(ctx, k); !errors.Is(err, store.ErrInvalidKey) {
			t.Errorf("Exists %q: got %v, want ErrInvalidKey", k, err)
		}
		if err := s.Delete(ctx, k); !errors.Is(err, store.ErrInvalidKey) {
			t.Errorf("Delete %q: got %v, want ErrInvalidKey", k, err)
		}
	}
	entries, err := os.ReadDir(parent)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if e.Name() != "root" {
			t.Errorf("escaping write created %q outside the store root", e.Name())
		}
	}
}

// TestResolveLeadingSlashIsRootRelative pins that leading slashes are
// ignored: "/abs" addresses the same value as "abs", inside the root.
func TestResolveLeadingSlashIsRootRelative(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)
	if err := s.Set(ctx, "/abs", []byte("one")); err != nil {
		t.Fatalf("Set(/abs): %v", err)
	}
	if err := s.Set(ctx, "//c/0", []byte("two")); err != nil {
		t.Fatalf("Set(//c/0): %v", err)
	}
	for k, want := range map[string]string{"abs": "one", "/abs": "one", "c/0": "two", "/c/0": "two"} {
		got, err := s.Get(ctx, k)
		if err != nil || string(got) != want {
			t.Errorf("Get(%q) = %q, %v; want %q, nil", k, got, err, want)
		}
	}
	if _, err := os.Stat(filepath.Join(s.root, "abs")); err != nil {
		t.Errorf("Set(/abs) did not land at root/abs: %v", err)
	}
}

// TestExistsFalseForDirectory checks that an intermediate directory (a key
// prefix, not a key) is not reported as an existing key — Get on it fails,
// so Exists must agree.
func TestExistsFalseForDirectory(t *testing.T) {
	ctx := context.Background()
	s := newStore(t)
	if err := s.Set(ctx, "dir/leaf", []byte("x")); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if ok, err := s.Exists(ctx, "dir"); err != nil || ok {
		t.Fatalf("Exists(dir) = %v, %v; want false, nil", ok, err)
	}
	if ok, err := s.Exists(ctx, "dir/leaf"); err != nil || !ok {
		t.Fatalf("Exists(dir/leaf) = %v, %v; want true, nil", ok, err)
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

// TestStoredFilesAreWorldReadable checks that published keys don't inherit
// os.CreateTemp's owner-only 0600 mode: a store written by one user must be
// readable by others (e.g. a web server serving the same directory).
func TestStoredFilesAreWorldReadable(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX permission bits are not meaningful on windows")
	}
	ctx := context.Background()
	s := newStore(t)
	if err := s.Set(ctx, "a/set", []byte("x")); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if ok, err := s.SetIfNotExists(ctx, "a/sine", []byte("x")); err != nil || !ok {
		t.Fatalf("SetIfNotExists: ok=%v err=%v", ok, err)
	}
	for _, k := range []string{"a/set", "a/sine"} {
		info, err := os.Stat(filepath.Join(s.root, filepath.FromSlash(k)))
		if err != nil {
			t.Fatalf("Stat %q: %v", k, err)
		}
		if perm := info.Mode().Perm(); perm&0o044 != 0o044 {
			t.Errorf("%q mode = %v, want group/other read bits (0o044) set", k, perm)
		}
	}
}
