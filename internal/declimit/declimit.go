// Package declimit hosts the shared decompression-output size cap consulted
// by the gzip, zlib, zstd and blosc codecs and the sharding index decoder.
//
// The cap defends against decompression bombs: a small hostile payload that
// expands to gigabytes. One shared value lets callers raise or disable the
// limit for every codec at once.
package declimit

import (
	"fmt"
	"math"
	"sync/atomic"
)

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

// CheckDeclaredSize validates a decompressor-declared output size — read
// from untrusted, attacker-controlled frame/header metadata — before it is
// used to size an allocation (e.g. `make([]byte, n)`). It rejects negative
// values, values that don't fit in an int (which would panic make() on
// build platforms where int is 32-bit), and values exceeding the configured
// cap. Codecs that learn their decoded size up front (zstd, blosc) must call
// this before allocating; codecs that decode incrementally (gzip, zlib)
// instead bound the read with io.LimitReader and check the result length.
func CheckDeclaredSize(n int64) error {
	if n < 0 {
		return fmt.Errorf("declared decoded size %d is negative", n)
	}
	if n > math.MaxInt {
		return fmt.Errorf("declared decoded size %d exceeds platform int range", n)
	}
	if max := Max(); max > 0 && n > max {
		return fmt.Errorf("declared decoded size %d exceeds configured maximum %d bytes", n, max)
	}
	return nil
}
