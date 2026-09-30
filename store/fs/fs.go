// Package fs provides a local-filesystem backend for the zarr Store interface.
//
// Keys are forward-slash separated and mapped to paths beneath a fixed root
// directory. All writes are atomic via temp-file + rename (Set) or
// temp-file + hard-link (SetIfNotExists), and durable by default (fsynced
// before publish; see WithNoSync to opt out for throughput-sensitive
// scratch use).
package fs

import (
	"context"
	"errors"
	"fmt"
	"io"
	stdfs "io/fs"
	"iter"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/rkm/zarr-go/store"
)

// tempPrefix names the reserved staging-file namespace Set/SetIfNotExists
// use for their temp files. Basenames matching this prefix are rejected as
// keys (see resolve) and filtered out of List results, so a crash-leaked
// temp file (before the following rename/link) never becomes visible as a
// phantom store key.
const tempPrefix = ".zarr-tmp-"

// Store is a local-filesystem implementation of store.Store and
// store.WritableStore. All keys are resolved relative to a fixed root.
type Store struct {
	root        string
	mmapEnabled bool
	noSync      bool
}

// testHookBeforeCommit, when non-nil, is invoked after a temp file has been
// fully written (and synced) but before the rename/link that publishes it.
// Used by tests to deterministically observe pre-commit state. Never set
// outside tests.
var testHookBeforeCommit func()

// New constructs a Store rooted at the given directory. The directory must
// already exist and be a directory.
func New(root string, opts ...Option) (*Store, error) {
	info, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("fs.New: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("fs.New: root %q is not a directory", root)
	}
	s := &Store{root: root}
	for _, opt := range opts {
		opt(s)
	}
	return s, nil
}

// MmapEnabled reports whether GetMmap is available on this Store. When false,
// the Store still satisfies the MmapStore interface but GetMmap falls back to
// Get + a no-op release.
func (s *Store) MmapEnabled() bool { return s.mmapEnabled }

// resolve translates a forward-slash key into an absolute path under the root,
// rejecting any key that contains a ".." segment or a segment in the
// reserved temp-file namespace.
func (s *Store) resolve(key string) (string, error) {
	if key == "" {
		return s.root, nil
	}
	// Reject "..": split on '/' so we catch it regardless of OS separator.
	for _, seg := range strings.Split(key, "/") {
		if seg == ".." {
			return "", fmt.Errorf("fs: key %q contains forbidden \"..\" segment", key)
		}
		if strings.HasPrefix(seg, tempPrefix) {
			return "", fmt.Errorf("fs: key %q contains reserved segment %q (prefix %q is reserved for staging files)", key, seg, tempPrefix)
		}
	}
	return filepath.Join(s.root, filepath.FromSlash(key)), nil
}

// Get returns the full value at key.
func (s *Store) Get(ctx context.Context, key string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	path, err := s.resolve(key)
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, stdfs.ErrNotExist) {
			return nil, store.ErrKeyNotFound
		}
		return nil, err
	}
	return b, nil
}

// GetRange returns up to n bytes starting at off from the value at key, per
// the store.Store GetRange contract.
func (s *Store) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if off < 0 || n < 0 {
		return nil, fmt.Errorf("fs: GetRange(off=%d, n=%d): %w", off, n, store.ErrInvalidRange)
	}
	path, err := s.resolve(key)
	if err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_RDONLY, 0)
	if err != nil {
		if errors.Is(err, stdfs.ErrNotExist) {
			return nil, store.ErrKeyNotFound
		}
		return nil, err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := info.Size()
	if off >= size {
		return []byte{}, nil
	}
	// Compare against the remaining length rather than off+n, which can
	// overflow int64 for n near MaxInt64 and skip the clamp.
	want := n
	if want > size-off {
		want = size - off
	}
	buf := make([]byte, want)
	read, err := f.ReadAt(buf, off)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, err
	}
	return buf[:read], nil
}

// GetSuffix returns the last n bytes of the value at key, per the
// store.SuffixReader contract.
func (s *Store) GetSuffix(ctx context.Context, key string, n int64) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if n < 0 {
		return nil, fmt.Errorf("fs: GetSuffix(n=%d): %w", n, store.ErrInvalidRange)
	}
	path, err := s.resolve(key)
	if err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_RDONLY, 0)
	if err != nil {
		if errors.Is(err, stdfs.ErrNotExist) {
			return nil, store.ErrKeyNotFound
		}
		return nil, err
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := info.Size()
	start := size - n
	if start < 0 {
		start = 0
	}
	want := size - start
	if want == 0 {
		return []byte{}, nil
	}
	buf := make([]byte, want)
	read, err := f.ReadAt(buf, start)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, err
	}
	return buf[:read], nil
}

// Set writes value at key, creating parent directories as needed. The write
// is atomic: data is written to a sibling temp file (fsynced by default),
// then renamed into place, then the parent directory is fsynced so the
// rename itself is durable. See WithNoSync to skip the fsyncs.
func (s *Store) Set(ctx context.Context, key string, value []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	path, err := s.resolve(key)
	if err != nil {
		return err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, tempPrefix+"*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	if err := s.writeAndCommit(tmp, tmpPath, value); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	if !s.noSync {
		if err := syncDir(dir); err != nil {
			return err
		}
	}
	return nil
}

// writeAndCommit writes value to tmp, syncs and closes it (unless noSync),
// and invokes testHookBeforeCommit — the shared prefix of Set and
// SetIfNotExists between "temp file created" and "temp file published".
// Removes the temp file and returns the error on any failure.
func (s *Store) writeAndCommit(tmp *os.File, tmpPath string, value []byte) error {
	// os.CreateTemp creates files 0600; widen to the usual 0644 so the
	// published key isn't owner-only.
	if err := tmp.Chmod(0o644); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
		return err
	}
	if _, err := tmp.Write(value); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
		return err
	}
	if !s.noSync {
		if err := tmp.Sync(); err != nil {
			_ = tmp.Close()
			_ = os.Remove(tmpPath)
			return err
		}
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	if testHookBeforeCommit != nil {
		testHookBeforeCommit()
	}
	return nil
}

// syncDir fsyncs a directory so a preceding rename/link within it is
// durable. A no-op on Windows, which doesn't support syncing directory
// handles.
func syncDir(dir string) error {
	if runtime.GOOS == "windows" {
		return nil
	}
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}

// SetIfNotExists writes value at key only if the key does not already exist.
// Returns false (without error) if the key was already present.
//
// Unlike Set, this writes the full content to a temp file first and
// publishes it via a hard link (not a direct O_EXCL write), so a concurrent
// reader can never observe a partially-written file at key, and a crash
// between the temp write and the link leaves no trace at key at all (the
// pre-fix O_EXCL approach could leave a permanently-empty file at key,
// wedging it as "already exists" forever). Requires the store's filesystem
// to support hard links within a directory (true for essentially all
// POSIX/NTFS filesystems; not FAT/exFAT).
func (s *Store) SetIfNotExists(ctx context.Context, key string, value []byte) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	path, err := s.resolve(key)
	if err != nil {
		return false, err
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return false, err
	}
	tmp, err := os.CreateTemp(dir, tempPrefix+"*")
	if err != nil {
		return false, err
	}
	tmpPath := tmp.Name()
	if err := s.writeAndCommit(tmp, tmpPath, value); err != nil {
		return false, err
	}
	defer os.Remove(tmpPath) // Link doesn't consume the source; always clean it up.

	if err := os.Link(tmpPath, path); err != nil {
		if errors.Is(err, stdfs.ErrExist) {
			return false, nil
		}
		return false, err
	}
	if !s.noSync {
		if err := syncDir(dir); err != nil {
			return true, err
		}
	}
	return true, nil
}

// Delete removes the value at key. Missing keys are not an error.
func (s *Store) Delete(ctx context.Context, key string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	path, err := s.resolve(key)
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !errors.Is(err, stdfs.ErrNotExist) {
		return err
	}
	return nil
}

// Exists reports whether key is present.
func (s *Store) Exists(ctx context.Context, key string) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	path, err := s.resolve(key)
	if err != nil {
		return false, err
	}
	if _, err := os.Stat(path); err != nil {
		if errors.Is(err, stdfs.ErrNotExist) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// List yields every key for which strings.HasPrefix(key, prefix) holds,
// per the store.Store List contract (pure string-prefix, not path-prefix —
// a prefix that isn't a complete path segment, e.g. "foo/ba", still matches
// "foo/bar"). Reserved staging-file entries (see tempPrefix) are never
// yielded.
//
// Implementation: starts the filesystem walk from the parent directory of
// prefix's last "/"-separated segment (an optimization — the walk needs
// only ever look inside directories whose path is a prefix of, or is
// prefixed by, that starting point) and prunes subtrees during the walk
// that can't possibly contain a match, so the full store is only walked
// when the prefix supplies no path segment at all (e.g. "" or a single
// top-level partial name).
func (s *Store) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		dirKey := ""
		if idx := strings.LastIndex(prefix, "/"); idx >= 0 {
			dirKey = prefix[:idx]
		}
		base, err := s.resolve(dirKey)
		if err != nil {
			yield("", err)
			return
		}
		info, err := os.Stat(base)
		if err != nil {
			if errors.Is(err, stdfs.ErrNotExist) {
				return
			}
			yield("", err)
			return
		}
		if !info.IsDir() {
			return
		}

		walkErr := filepath.WalkDir(base, func(path string, d stdfs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if cerr := ctx.Err(); cerr != nil {
				return cerr
			}
			rel, rerr := filepath.Rel(s.root, path)
			if rerr != nil {
				return rerr
			}
			key := filepath.ToSlash(rel)
			if d.IsDir() {
				if key == "." {
					return nil
				}
				if strings.HasPrefix(key+"/", prefix) || strings.HasPrefix(prefix, key+"/") {
					return nil
				}
				return filepath.SkipDir
			}
			if strings.HasPrefix(d.Name(), tempPrefix) {
				return nil
			}
			if !strings.HasPrefix(key, prefix) {
				return nil
			}
			if !yield(key, nil) {
				return filepath.SkipAll
			}
			return nil
		})
		if walkErr != nil && !errors.Is(walkErr, filepath.SkipAll) {
			yield("", walkErr)
		}
	}
}

// Compile-time interface assertions.
var (
	_ store.Store         = (*Store)(nil)
	_ store.WritableStore = (*Store)(nil)
	_ store.SuffixReader  = (*Store)(nil)
)
