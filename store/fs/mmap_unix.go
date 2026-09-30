//go:build linux || darwin || freebsd || netbsd || openbsd || dragonfly

package fs

import (
	"context"
	"errors"
	"fmt"
	stdfs "io/fs"
	"os"
	"sync"
	"syscall"

	"github.com/exergy-dev/zarr-go/store"
)

// noopRelease is a non-nil release callback for paths where no resources are
// held (empty file, fallback path). It's safe to defer unconditionally.
func noopRelease() {}

// GetMmap returns the contents of key via mmap. The returned slice is valid
// until release() is called; callers MUST call release() to unmap it. The
// underlying file is closed before GetMmap returns (the mapping outlives
// it). release is always non-nil on success.
//
// When the Store was constructed without WithMmap(), this falls back to Get
// and returns a no-op release so callers can probe MmapStore unconditionally.
//
// Empty files are mapped as a zero-length slice (mmap of size 0 is invalid on
// most platforms, so we just open+close and return an empty slice).
func (s *Store) GetMmap(ctx context.Context, key string) ([]byte, func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, noopRelease, err
	}
	if !s.mmapEnabled {
		b, err := s.Get(ctx, key)
		if err != nil {
			return nil, noopRelease, err
		}
		return b, noopRelease, nil
	}
	path, err := s.resolve(key)
	if err != nil {
		return nil, noopRelease, err
	}
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, stdfs.ErrNotExist) {
			return nil, noopRelease, store.ErrKeyNotFound
		}
		return nil, noopRelease, err
	}
	// The mapping stays valid after its fd is closed, so close on every
	// path, success included: live mappings must not pin fds.
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, noopRelease, err
	}
	size := info.Size()
	if size == 0 {
		// mmap of size 0 is invalid on Linux; return empty.
		return []byte{}, noopRelease, nil
	}
	if size > int64(^uint(0)>>1) {
		return nil, noopRelease, fmt.Errorf("fs.GetMmap: file %q too large for mmap (%d bytes)", path, size)
	}
	// The standard library's syscall.Mmap exists on every platform in this
	// file's build constraint, so this needs no golang.org/x/sys.
	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, noopRelease, fmt.Errorf("fs.GetMmap: mmap %q: %w", path, err)
	}
	// Wrap the unmap so a buggy double-call from the engine cannot Munmap
	// already-freed pages (possibly reused by a later mapping) — silent UB
	// on Linux otherwise.
	release := sync.OnceFunc(func() {
		_ = syscall.Munmap(data)
	})
	return data, release, nil
}

// Compile-time interface assertion for the unix build.
var _ store.MmapStore = (*Store)(nil)
