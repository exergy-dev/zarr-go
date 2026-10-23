package zarr

import "runtime"

type arrayOptions struct {
	concurrency   int
	ioConcurrency int
	rangeGap      int64
}

func defaultOptions() arrayOptions {
	n := runtime.NumCPU()
	return arrayOptions{
		concurrency:   n,
		ioConcurrency: 4 * n,
		rangeGap:      4 << 20, // 4 MiB
	}
}

// OpenOption customizes Open / OpenArray. It is also a CreateOption, so the
// same options configure the Array that CreateArray returns.
type OpenOption func(*arrayOptions)

func (f OpenOption) applyCreate(o *createOptions) {
	o.arrayOnly = "OpenOption"
	f(&o.openOpts)
}

// WithConcurrency caps the number of chunks being decoded (Read) or
// encoded (Write) at once. The default is the number of CPUs.
func WithConcurrency(n int) OpenOption {
	return func(o *arrayOptions) {
		if n > 0 {
			o.concurrency = n
		}
	}
}

// WithIOConcurrency caps the number of in-flight store requests a Read or
// Write issues. The default is four times the number of CPUs.
func WithIOConcurrency(n int) OpenOption {
	return func(o *arrayOptions) {
		if n > 0 {
			o.ioConcurrency = n
		}
	}
}

// WithRangeGap sets the maximum gap (in bytes) between adjacent ranges that
// will be coalesced into a single GetRange call. Used by the sharded
// partial-decode path: inner chunks intersecting a read's region are usually
// NOT contiguous in the shard body (unneeded inner chunks lie between needed
// rows), so paying for a few KB of extra payload is usually cheaper than
// paying for the latency of another round trip on a high-RTT object store.
// 0 disables coalescing beyond exact adjacency.
func WithRangeGap(b int64) OpenOption {
	return func(o *arrayOptions) {
		if b >= 0 {
			o.rangeGap = b
		}
	}
}
