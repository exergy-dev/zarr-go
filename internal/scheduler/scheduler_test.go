package scheduler

import (
	"math"
	"testing"
)

func TestCoalesceRangesNoMerge(t *testing.T) {
	in := []Range{{0, 10}, {100, 10}, {200, 10}}
	out := CoalesceRanges(in, 5)
	if len(out) != 3 {
		t.Fatalf("want 3 merged ranges, got %d", len(out))
	}
}

func TestCoalesceRangesAdjacent(t *testing.T) {
	in := []Range{{0, 10}, {10, 5}, {15, 5}}
	out := CoalesceRanges(in, 0)
	if len(out) != 1 {
		t.Fatalf("want 1 merged range, got %d", len(out))
	}
	if out[0].Off != 0 || out[0].N != 20 {
		t.Fatalf("merged span want (0,20), got (%d,%d)", out[0].Off, out[0].N)
	}
	if len(out[0].Members) != 3 {
		t.Fatalf("want 3 members, got %d", len(out[0].Members))
	}
}

func TestCoalesceRangesGap(t *testing.T) {
	in := []Range{{0, 10}, {12, 10}}
	if got := CoalesceRanges(in, 1); len(got) != 2 {
		t.Fatalf("gap=1 should not merge gap of 2: got %d", len(got))
	}
	if got := CoalesceRanges(in, 2); len(got) != 1 {
		t.Fatalf("gap=2 should merge gap of 2: got %d", len(got))
	}
}

func TestCoalesceRangesPreservesIndices(t *testing.T) {
	in := []Range{{200, 5}, {0, 5}, {100, 5}} // out of order
	out := CoalesceRanges(in, 0)
	if len(out) != 3 {
		t.Fatalf("want 3 ranges, got %d", len(out))
	}
	// Each merged range should have one member with the correct OriginalIndex.
	seen := map[int]bool{}
	for _, m := range out {
		for _, mem := range m.Members {
			seen[mem.OriginalIndex] = true
		}
	}
	for i := 0; i < 3; i++ {
		if !seen[i] {
			t.Fatalf("OriginalIndex %d missing", i)
		}
	}
}

// TestCoalesceRangesOverflow: a range whose Off+N overflows int64 must not
// wrap into a "fully contained" member that points past the merged span.
func TestCoalesceRangesOverflow(t *testing.T) {
	in := []Range{{0, 4}, {1000, math.MaxInt64}}
	out := CoalesceRanges(in, 4<<20)
	for _, m := range out {
		if m.N < 0 || m.N > math.MaxInt64-m.Off {
			t.Fatalf("merged range (%d,%d) wraps int64", m.Off, m.N)
		}
		for _, mem := range m.Members {
			if mem.OffInMerged > m.N {
				t.Fatalf("member %d starts at %d, past merged span of %d bytes",
					mem.OriginalIndex, mem.OffInMerged, m.N)
			}
		}
	}
}
