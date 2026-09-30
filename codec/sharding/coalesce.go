package sharding

import (
	"cmp"
	"math"
	"slices"
)

// byteRange is an (offset, length) request against the shard blob.
type byteRange struct {
	off, n int64
}

// mergedRange is one coalesced fetch covering a contiguous span. members maps
// each original byteRange (by index into the input slice) to its slice of the
// merged blob.
type mergedRange struct {
	off, n  int64
	members []rangeMember
}

// rangeMember binds an original request back to its slice within a
// mergedRange.
type rangeMember struct {
	idx         int   // index into the coalesceRanges input
	offInMerged int64 // offset of this member within the merged blob
	n           int64
}

// coalesceRanges merges ranges whose gap is <= gapBytes into a single
// mergedRange, so adjacent inner chunks can be fetched in one round-trip.
//
// off and n must be non-negative. Range ends saturate at math.MaxInt64
// rather than wrapping, so an off+n that overflows (e.g. from an unvalidated
// shard index) can't masquerade as a range contained in an earlier one.
// Callers must still bounds-check each member against the bytes fetched.
func coalesceRanges(in []byteRange, gapBytes int64) []mergedRange {
	if len(in) == 0 {
		return nil
	}
	type indexed struct {
		byteRange
		idx int
	}
	idx := make([]indexed, len(in))
	for i, r := range in {
		idx[i] = indexed{r, i}
	}
	slices.SortFunc(idx, func(a, b indexed) int { return cmp.Compare(a.off, b.off) })

	var out []mergedRange
	cur := mergedRange{off: idx[0].off, n: idx[0].n, members: []rangeMember{{idx: idx[0].idx, n: idx[0].n}}}
	for _, r := range idx[1:] {
		curEnd := rangeEnd(cur.off, cur.n)
		if r.off-curEnd > gapBytes {
			out = append(out, cur)
			cur = mergedRange{off: r.off, n: r.n, members: []rangeMember{{idx: r.idx, n: r.n}}}
			continue
		}
		// Extend cur to cover r (unless r is fully contained in it).
		if extra := rangeEnd(r.off, r.n) - curEnd; extra > 0 {
			cur.n += extra
		}
		cur.members = append(cur.members, rangeMember{idx: r.idx, offInMerged: r.off - cur.off, n: r.n})
	}
	return append(out, cur)
}

// rangeEnd returns off+n, saturating at math.MaxInt64 instead of wrapping.
func rangeEnd(off, n int64) int64 {
	if n > math.MaxInt64-off {
		return math.MaxInt64
	}
	return off + n
}
