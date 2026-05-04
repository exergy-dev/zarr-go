// Package bufpool provides sized sync.Pools of []byte keyed by power-of-two
// byte-size class. Above the largest class we fall back to ad-hoc allocation
// rather than panic.
package bufpool

import (
	"math/bits"
	"sync"
)

const (
	minClassLog = 12 // 4 KiB
	maxClassLog = 27 // 128 MiB
	numClasses  = maxClassLog - minClassLog + 1
)

var pools [numClasses]sync.Pool

func init() {
	for i := range pools {
		size := 1 << (minClassLog + i)
		pools[i].New = func() any {
			b := make([]byte, size)
			return &b
		}
	}
}

func classFor(n int) int {
	if n <= 1<<minClassLog {
		return 0
	}
	// smallest 2^k >= n
	k := bits.Len(uint(n - 1))
	return k - minClassLog
}

// Get returns a []byte with len == n. The underlying capacity is the next
// power-of-two class >= n. Use Put to return it to the pool.
//
// Buffers larger than 1<<maxClassLog (128 MiB) are allocated directly and
// dropped on Put.
func Get(n int) []byte {
	if n <= 0 {
		return nil
	}
	cls := classFor(n)
	if cls >= numClasses {
		return make([]byte, n)
	}
	b := pools[cls].Get().(*[]byte)
	return (*b)[:n]
}

// Put returns a buffer obtained from Get to its pool. The slice's capacity
// determines the destination class — slicing past cap before Put loses the
// buffer to GC, which is safe but defeats the pool.
func Put(b []byte) {
	if cap(b) < 1<<minClassLog {
		return
	}
	c := cap(b)
	if c > 1<<maxClassLog {
		return
	}
	// recover the class from cap (which is exactly a power of two for pooled bufs)
	k := bits.TrailingZeros(uint(c))
	cls := k - minClassLog
	if cls < 0 || cls >= numClasses {
		return
	}
	full := b[:c]
	pools[cls].Put(&full)
}
