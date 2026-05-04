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
		rangeGap:      1 << 20, // 1 MiB
	}
}

// OpenOption customizes Open / OpenArray.
type OpenOption func(*arrayOptions)

// WithConcurrency caps the number of decode workers.
func WithConcurrency(n int) OpenOption {
	return func(o *arrayOptions) {
		if n > 0 {
			o.concurrency = n
		}
	}
}

// WithIOConcurrency caps the number of in-flight store fetches.
func WithIOConcurrency(n int) OpenOption {
	return func(o *arrayOptions) {
		if n > 0 {
			o.ioConcurrency = n
		}
	}
}

// WithRangeGap sets the maximum gap (in bytes) between adjacent ranges that
// will be coalesced into a single GetRange call.
func WithRangeGap(b int64) OpenOption {
	return func(o *arrayOptions) {
		if b >= 0 {
			o.rangeGap = b
		}
	}
}
