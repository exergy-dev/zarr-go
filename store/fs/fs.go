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

	"github.com/exergy-dev/zarr-go/store"
)

// tempPrefix names the reserved staging-file namespace Set/SetIfNotExists
// use for their temp files. Basenames matching this prefix are rejected as
// keys (see resolve) and filtered out of List results, so a crash-leaked
// temp file (before the following rename/link) never becomes visible as a
// phantom store key.
const tempPrefix = ".zarr-tmp-"

// Store is a local-filesystem implementation of store.Store and
// store.Writer. All keys are resolved relative to a fixed root.
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

// MmapEnabled reports whether GetMmap memory-maps (see WithMmap). When false,
// GetMmap falls back to Get with a no-op release.
func (s *Store) MmapEnabled() bool { return s.mmapEnabled }

// resolve translates a forward-slash key into an absolute path under the root,
// rejecting any key that contains a ".." segment, a segment in the reserved
// temp-file namespace, or anything else that isn't a local path on this OS
// (e.g. `..\x` or a volume name on windows). Leading slashes are ignored, so
// "/a" and "a" name the same key.
func (s *Store) resolve(key string) (string, error) {
	key = strings.TrimLeft(key, "/")
	if key == "" {
		return s.root, nil
	}
	// Split on '/' so ".." is caught regardless of OS separator.
	for _, seg := range strings.Split(key, "/") {
		if seg == ".." {
			return "", fmt.Errorf("fs: key %q contains a \"..\" segment: %w", key, store.ErrInvalidKey)
		}
		if strings.HasPrefix(seg, tempPrefix) {
			return "", fmt.Errorf("fs: key %q contains segment %q (prefix %q is reserved for staging files): %w", key, seg, tempPrefix, store.ErrInvalidKey)
		}
	}
	rel := filepath.FromSlash(key)
	if !filepath.IsLocal(rel) {
		return "", fmt.Errorf("fs: key %q is not a local path beneath the store root: %w", key, store.ErrInvalidKey)
	}
	return filepath.Join(s.root, rel), nil
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
// store.Store GetSuffix contract.
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

// writeAndCommit writes value to tmp, syncs it (unless noSync), closes it,
// and invokes testHookBeforeCommit. On any failure it removes the temp file
// and returns the error.
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
// The full content goes to a temp file first and is published via a hard
// link, which fails if key exists. A concurrent reader never observes a
// partially-written file, and a crash before the link leaves nothing at key
// (a direct O_EXCL write could leave an empty file that then reads as
// "already exists" forever). Requires hard-link support within a directory
// (POSIX filesystems and NTFS; not FAT/exFAT).
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

// Exists reports whether key is present. A directory (a prefix of other
// keys, not a key itself) is not present.
func (s *Store) Exists(ctx context.Context, key string) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	path, err := s.resolve(key)
	if err != nil {
		return false, err
	}
	info, err := os.Stat(path)
	if err != nil {
		if errors.Is(err, stdfs.ErrNotExist) {
			return false, nil
		}
		return false, err
	}
	return !info.IsDir(), nil
}

// List yields every key for which strings.HasPrefix(key, prefix) holds,
// per the store.Store List contract (a string prefix, not a path prefix:
// "foo/ba" matches "foo/bar"). Staging files (see tempPrefix) are never
// yielded.
//
// The walk starts at the directory named by prefix up to its last "/" and
// prunes subtrees that can't contain a match.
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

// Compile-time interface assertion.
var _ store.Writer = (*Store)(nil)
