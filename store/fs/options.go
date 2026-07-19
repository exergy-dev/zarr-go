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

// WithNoSync disables fsync-before-rename/link on Set and SetIfNotExists.
// By default, writes are durable: the temp file is fsynced before the
// rename/link that publishes it, and the parent directory is fsynced after
// (so the rename itself survives a crash on filesystems that require it).
// Without WithNoSync, "atomic" only means "never observed partially
// written" — a power loss can still lose an acknowledged write. Pass this
// for throughput-sensitive scratch/test use where that durability isn't
// needed.
func WithNoSync() Option {
	return func(s *Store) {
		s.noSync = true
	}
}
