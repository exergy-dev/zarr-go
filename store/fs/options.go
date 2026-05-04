package fs

// Option configures a Store at construction time.
type Option func(*Store)

// WithMmap enables the GetMmap fast path on this Store. When enabled, the
// Store implements store.MmapStore and the engine can read chunk bytes
// directly from a memory-mapped file region without an intermediate copy.
//
// This is most beneficial for large, uncompressed chunks on local NVMe-backed
// storage. It is opt-in because mmap behaves surprisingly on truncation or
// non-local filesystems.
func WithMmap() Option {
	return func(s *Store) {
		s.mmapEnabled = true
	}
}
