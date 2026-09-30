package fs

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/exergy-dev/zarr-go/store"
)

func TestMmapGet(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir(), WithMmap())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if !s.MmapEnabled() {
		t.Fatal("MmapEnabled: got false, want true")
	}
	want := bytes.Repeat([]byte("hello-zarr"), 4096) // 40 KiB
	if err := s.Set(ctx, "blob", want); err != nil {
		t.Fatalf("Set: %v", err)
	}
	data, release, err := s.GetMmap(ctx, "blob")
	if err != nil {
		t.Fatalf("GetMmap: %v", err)
	}
	if release == nil {
		t.Fatal("GetMmap: release callback is nil")
	}
	defer release()
	if !bytes.Equal(data, want) {
		t.Fatalf("GetMmap: bytes mismatch (len got=%d want=%d)", len(data), len(want))
	}
}

func TestMmapMissingKey(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir(), WithMmap())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, release, err := s.GetMmap(ctx, "no/such/key")
	if !errors.Is(err, store.ErrKeyNotFound) {
		t.Fatalf("GetMmap missing: err=%v, want store.ErrKeyNotFound", err)
	}
	if release == nil {
		t.Fatal("GetMmap: release should be non-nil even on error so callers can defer it")
	}
	release() // must not panic
}

func TestMmapDisabledFallsBackToGet(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir()) // no WithMmap()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if s.MmapEnabled() {
		t.Fatal("MmapEnabled: got true, want false")
	}
	want := []byte("xyz")
	if err := s.Set(ctx, "k", want); err != nil {
		t.Fatalf("Set: %v", err)
	}
	data, release, err := s.GetMmap(ctx, "k")
	if err != nil {
		t.Fatalf("GetMmap: %v", err)
	}
	defer release()
	if !bytes.Equal(data, want) {
		t.Fatalf("GetMmap: got %q, want %q", data, want)
	}
}

func TestMmapEmptyFile(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir(), WithMmap())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := s.Set(ctx, "empty", []byte{}); err != nil {
		t.Fatalf("Set: %v", err)
	}
	data, release, err := s.GetMmap(ctx, "empty")
	if err != nil {
		t.Fatalf("GetMmap empty: %v", err)
	}
	defer release()
	if len(data) != 0 {
		t.Fatalf("GetMmap empty: got %d bytes, want 0", len(data))
	}
}

func TestMmapSatisfiesInterface(t *testing.T) {
	s, err := New(t.TempDir(), WithMmap())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	var _ store.MmapStore = s
}

// TestMmapDoesNotHoldFd checks that a live mapping does not pin an open file
// descriptor: the mapping outlives the fd, so GetMmap closes it immediately
// and release only unmaps. Holding the fd until release would let a caller
// with many live mappings exhaust the fd limit.
func TestMmapDoesNotHoldFd(t *testing.T) {
	const fdDir = "/proc/self/fd"
	if _, err := os.Stat(fdDir); err != nil {
		t.Skipf("%s unavailable: %v", fdDir, err)
	}
	ctx := context.Background()
	s, err := New(t.TempDir(), WithMmap())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	want := bytes.Repeat([]byte("m"), 4096)
	if err := s.Set(ctx, "k", want); err != nil {
		t.Fatalf("Set: %v", err)
	}
	path, err := s.resolve("k")
	if err != nil {
		t.Fatal(err)
	}
	data, release, err := s.GetMmap(ctx, "k")
	if err != nil {
		t.Fatalf("GetMmap: %v", err)
	}
	defer release()

	entries, err := os.ReadDir(fdDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if target, err := os.Readlink(filepath.Join(fdDir, e.Name())); err == nil && target == path {
			t.Errorf("fd %s still open on %s while mapping is live", e.Name(), path)
		}
	}
	if !bytes.Equal(data, want) {
		t.Fatalf("mapping unreadable after fd close: len got=%d want=%d", len(data), len(want))
	}
}

// TestMmapNoLeakLoop opens many keys via mmap in a tight loop. If release
// fails to munmap+close, we'd hit "too many open files" within a few hundred
// iterations on a default ulimit. The check is implicit: the loop must
// complete without error.
func TestMmapNoLeakLoop(t *testing.T) {
	ctx := context.Background()
	s, err := New(t.TempDir(), WithMmap())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	payload := bytes.Repeat([]byte("x"), 4096)
	if err := s.Set(ctx, "k", payload); err != nil {
		t.Fatalf("Set: %v", err)
	}
	// 2000 iterations far exceeds typical ulimit -n if we leak fds.
	for i := 0; i < 2000; i++ {
		data, release, err := s.GetMmap(ctx, "k")
		if err != nil {
			t.Fatalf("iter %d GetMmap: %v", i, err)
		}
		if len(data) != len(payload) {
			t.Fatalf("iter %d size mismatch: %d vs %d", i, len(data), len(payload))
		}
		release()
	}
}
