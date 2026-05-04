//go:build !(linux || darwin || freebsd || netbsd || openbsd || dragonfly)

package fs

import (
	"context"

	"github.com/rkm/zarr-go/store"
)

// noopRelease is a non-nil release callback for the fallback path; nothing
// is held to release.
func noopRelease() {}

// GetMmap on non-unix platforms falls back to Get + a no-op release. The
// MmapStore interface is still satisfied so callers can probe uniformly.
func (s *Store) GetMmap(ctx context.Context, key string) ([]byte, func(), error) {
	b, err := s.Get(ctx, key)
	if err != nil {
		return nil, noopRelease, err
	}
	return b, noopRelease, nil
}

// Compile-time interface assertion for the fallback build.
var _ store.MmapStore = (*Store)(nil)
