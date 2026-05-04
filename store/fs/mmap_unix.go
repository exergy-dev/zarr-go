//go:build linux || darwin || freebsd || netbsd || openbsd || dragonfly

package fs

import (
	"context"
	"errors"
	"fmt"
	stdfs "io/fs"
	"os"
	"sync"

	"github.com/rkm/zarr-go/store"
	"golang.org/x/sys/unix"
)

// noopRelease is a non-nil release callback for paths where no resources are
// held (empty file, fallback path). It's safe to defer unconditionally.
func noopRelease() {}

// GetMmap returns the contents of key via mmap. The returned slice is valid
// until release() is called; callers MUST call release() to unmap and close
// the underlying file. release is always non-nil on success.
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
	info, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, noopRelease, err
	}
	size := info.Size()
	if size == 0 {
		// mmap of size 0 is invalid on Linux; close immediately and return empty.
		_ = f.Close()
		return []byte{}, noopRelease, nil
	}
	if size > int64(^uint(0)>>1) {
		_ = f.Close()
		return nil, noopRelease, fmt.Errorf("fs.GetMmap: file %q too large for mmap (%d bytes)", path, size)
	}
	data, err := unix.Mmap(int(f.Fd()), 0, int(size), unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		_ = f.Close()
		return nil, noopRelease, fmt.Errorf("fs.GetMmap: mmap %q: %w", path, err)
	}
	// Wrap the unmap+close pair so a buggy double-call from the engine
	// cannot Munmap freed pages or Close a closed fd — both are silent UB
	// on Linux otherwise.
	release := sync.OnceFunc(func() {
		_ = unix.Munmap(data)
		_ = f.Close()
	})
	return data, release, nil
}

// Compile-time interface assertion for the unix build.
var _ store.MmapStore = (*Store)(nil)
