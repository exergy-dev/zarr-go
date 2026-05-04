// Package fs provides a local-filesystem backend for the zarr Store interface.
//
// Keys are forward-slash separated and mapped to paths beneath a fixed root
// directory. All writes are atomic via temp-file + rename.
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
	"strings"

	"github.com/rkm/zarr-go/store"
)

// Store is a local-filesystem implementation of store.Store and
// store.WritableStore. All keys are resolved relative to a fixed root.
type Store struct {
	root string
}

// New constructs a Store rooted at the given directory. The directory must
// already exist and be a directory.
func New(root string) (*Store, error) {
	info, err := os.Stat(root)
	if err != nil {
		return nil, fmt.Errorf("fs.New: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("fs.New: root %q is not a directory", root)
	}
	return &Store{root: root}, nil
}

// resolve translates a forward-slash key into an absolute path under the root,
// rejecting any key that contains a ".." segment.
func (s *Store) resolve(key string) (string, error) {
	if key == "" {
		return s.root, nil
	}
	// Reject "..": split on '/' so we catch it regardless of OS separator.
	for _, seg := range strings.Split(key, "/") {
		if seg == ".." {
			return "", fmt.Errorf("fs: key %q contains forbidden \"..\" segment", key)
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

// GetRange returns up to n bytes starting at off from the value at key.
// If off+n exceeds the file size, the returned slice is clamped to whatever
// bytes are available at or after off.
func (s *Store) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if off < 0 || n < 0 {
		return nil, fmt.Errorf("fs.GetRange: negative off/n (off=%d n=%d)", off, n)
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
	want := n
	if off+want > size {
		want = size - off
	}
	buf := make([]byte, want)
	read, err := f.ReadAt(buf, off)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, err
	}
	return buf[:read], nil
}

// Set writes value at key, creating parent directories as needed. The write
// is atomic: data is written to a sibling temp file and then renamed.
func (s *Store) Set(ctx context.Context, key string, value []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	path, err := s.resolve(key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, value, 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		// Best-effort cleanup of the temp file on rename failure.
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

// SetIfNotExists writes value at key only if the key does not already exist.
// Returns false (without error) if the key was already present.
func (s *Store) SetIfNotExists(ctx context.Context, key string, value []byte) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	path, err := s.resolve(key)
	if err != nil {
		return false, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return false, err
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		if errors.Is(err, stdfs.ErrExist) {
			return false, nil
		}
		return false, err
	}
	if _, werr := f.Write(value); werr != nil {
		_ = f.Close()
		_ = os.Remove(path)
		return false, werr
	}
	if cerr := f.Close(); cerr != nil {
		return false, cerr
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

// List yields all file keys under prefix (recursively). If the prefix does not
// resolve to an existing directory, the iterator yields nothing without error.
func (s *Store) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		base, err := s.resolve(prefix)
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
		// If prefix points at a regular file, yield just that key.
		if !info.IsDir() {
			rel, rerr := filepath.Rel(s.root, base)
			if rerr != nil {
				yield("", rerr)
				return
			}
			yield(filepath.ToSlash(rel), nil)
			return
		}

		walkErr := filepath.WalkDir(base, func(path string, d stdfs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if cerr := ctx.Err(); cerr != nil {
				return cerr
			}
			if d.IsDir() {
				return nil
			}
			rel, rerr := filepath.Rel(s.root, path)
			if rerr != nil {
				return rerr
			}
			if !yield(filepath.ToSlash(rel), nil) {
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
)
