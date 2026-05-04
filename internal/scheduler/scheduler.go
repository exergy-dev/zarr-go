// Package scheduler provides bounded I/O and decode worker pools used by the
// array engine. It also implements range coalescing so that adjacent inner
// chunks of a sharded blob can be fetched in one round-trip.
package scheduler

import (
	"context"
	"sort"
	"sync"
)

// Range describes a (offset, length) request against a single key.
type Range struct {
	Off, N int64
}

// MergedRange is one coalesced fetch covering a contiguous span. Members lists
// the original Range entries (with their indices into the input slice) that
// fall inside this merged span.
type MergedRange struct {
	Off, N  int64
	Members []RangeMember
}

// RangeMember binds an original request back to its slice within MergedRange.
type RangeMember struct {
	OriginalIndex int
	OffInMerged   int64 // offset of this member within the merged blob
	N             int64
}

// CoalesceRanges merges adjacent ranges whose gap is <= gapBytes into a single
// MergedRange. Input is taken by value; output has stable mapping back to
// the input via OriginalIndex.
func CoalesceRanges(in []Range, gapBytes int64) []MergedRange {
	if len(in) == 0 {
		return nil
	}
	type indexed struct {
		Range
		Idx int
	}
	idx := make([]indexed, len(in))
	for i, r := range in {
		idx[i] = indexed{r, i}
	}
	sort.Slice(idx, func(i, j int) bool { return idx[i].Off < idx[j].Off })

	out := make([]MergedRange, 0)
	cur := MergedRange{Off: idx[0].Off, N: idx[0].N}
	cur.Members = append(cur.Members, RangeMember{
		OriginalIndex: idx[0].Idx, OffInMerged: 0, N: idx[0].N,
	})
	for k := 1; k < len(idx); k++ {
		r := idx[k]
		curEnd := cur.Off + cur.N
		gap := r.Off - curEnd
		if gap <= gapBytes {
			// Extend cur to cover r.
			extra := (r.Off + r.N) - curEnd
			if extra < 0 {
				// Overlap; r is fully contained.
				cur.Members = append(cur.Members, RangeMember{
					OriginalIndex: r.Idx, OffInMerged: r.Off - cur.Off, N: r.N,
				})
				continue
			}
			cur.N += extra
			cur.Members = append(cur.Members, RangeMember{
				OriginalIndex: r.Idx, OffInMerged: r.Off - cur.Off, N: r.N,
			})
		} else {
			out = append(out, cur)
			cur = MergedRange{Off: r.Off, N: r.N, Members: []RangeMember{{
				OriginalIndex: r.Idx, OffInMerged: 0, N: r.N,
			}}}
		}
	}
	out = append(out, cur)
	return out
}

// Pool runs jobs on a bounded number of workers.
type Pool struct {
	work chan func()
	wg   sync.WaitGroup
}

// NewPool constructs a worker pool with workers goroutines.
func NewPool(workers int) *Pool {
	if workers < 1 {
		workers = 1
	}
	p := &Pool{work: make(chan func(), workers*2)}
	p.wg.Add(workers)
	for i := 0; i < workers; i++ {
		go func() {
			defer p.wg.Done()
			for f := range p.work {
				f()
			}
		}()
	}
	return p
}

// Submit enqueues a job. Returns ctx.Err() if ctx is cancelled before the job
// can be enqueued.
func (p *Pool) Submit(ctx context.Context, f func()) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case p.work <- f:
		return nil
	}
}

// Close signals workers to exit and waits for them.
func (p *Pool) Close() {
	close(p.work)
	p.wg.Wait()
}
