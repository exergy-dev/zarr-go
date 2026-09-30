// Package scheduler implements range coalescing so that adjacent inner
// chunks of a sharded blob can be fetched in one round-trip.
package scheduler

import (
	"math"
	"sort"
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
//
// Off and N must be non-negative. Range ends saturate at math.MaxInt64
// rather than wrapping, so an Off+N that overflows (e.g. from an unvalidated
// shard index) can't masquerade as a range contained in an earlier one.
// Callers must still bounds-check each member against the bytes fetched.
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
		curEnd := rangeEnd(cur.Off, cur.N)
		gap := r.Off - curEnd
		if gap <= gapBytes {
			// Extend cur to cover r.
			extra := rangeEnd(r.Off, r.N) - curEnd
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

// rangeEnd returns off+n, saturating at math.MaxInt64 instead of wrapping.
func rangeEnd(off, n int64) int64 {
	if n > math.MaxInt64-off {
		return math.MaxInt64
	}
	return off + n
}
