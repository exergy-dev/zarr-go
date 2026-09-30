package fs

// Option configures a Store at construction time.
type Option func(*Store)

// WithMmap makes GetMmap memory-map files, so the engine can read chunk
// bytes without an intermediate copy. Without it, GetMmap falls back to Get.
//
// This is most beneficial for large, uncompressed chunks on local NVMe-backed
// storage. It is opt-in because mmap behaves surprisingly on truncation or
// non-local filesystems.
func WithMmap() Option {
	return func(s *Store) {
		s.mmapEnabled = true
	}
}

// WithNoSync disables the fsyncs on Set and SetIfNotExists. By default,
// writes are durable: the temp file is fsynced before the rename/link that
// publishes it, and the parent directory is fsynced after (so the rename
// itself survives a crash). With WithNoSync, writes are still atomic (never
// observed partially written) but a power loss can lose an acknowledged
// write. Use it for scratch or test data.
func WithNoSync() Option {
	return func(s *Store) {
		s.noSync = true
	}
}
