// Package declimit hosts the shared decompression-output size cap consulted
// by the gzip / zlib / zstd codecs.
//
// The cap defends against decompression bombs: hostile metadata can declare
// a small encoded chunk that expands to gigabytes. A single shared default
// keeps all three codecs in sync and lets callers raise/disable the limit
// once at process start.
package declimit

import "sync/atomic"

// defaultMax is 1 GiB; large enough for any realistic chunk, small enough
// that an OOM is replaced by an error.
const defaultMax = 1 << 30

var maxBytes atomic.Int64

func init() { maxBytes.Store(defaultMax) }

// Max returns the current cap. 0 means no cap.
func Max() int64 { return maxBytes.Load() }

// SetMax replaces the cap atomically. Pass 0 to disable. Safe to call
// concurrently; the new value applies to subsequent codec invocations.
func SetMax(n int64) { maxBytes.Store(n) }
