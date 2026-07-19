package ndcopy

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"
)

// naiveCopy is an independent, deliberately unoptimized reference
// implementation: walk every destination element in row-major order and
// copy the corresponding source element one at a time. Copy's fast
// (contiguous-merge) path is checked against this for many random
// shapes/regions; if they ever disagree, the merge arithmetic is wrong.
func naiveCopy(dst []byte, dstShape []int64, dstRegion []Region, src []byte, srcShape []int64, srcRegion []Region, elemSize int) []byte {
	out := append([]byte(nil), dst...)
	rank := len(srcShape)
	if rank == 0 {
		copy(out[:elemSize], src[:elemSize])
		return out
	}
	srcStrides := make([]int64, rank)
	dstStrides := make([]int64, rank)
	srcStrides[rank-1] = 1
	dstStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		srcStrides[i] = srcStrides[i+1] * srcShape[i+1]
		dstStrides[i] = dstStrides[i+1] * dstShape[i+1]
	}
	regionLen := make([]int64, rank)
	for i := range regionLen {
		regionLen[i] = dstRegion[i].Length()
	}
	total := int64(1)
	for _, n := range regionLen {
		total *= n
	}
	idx := make([]int64, rank)
	for k := int64(0); k < total; k++ {
		var srcEl, dstEl int64
		for d := 0; d < rank; d++ {
			srcEl += (srcRegion[d].Start + idx[d]*srcRegion[d].Step) * srcStrides[d]
			dstEl += (dstRegion[d].Start + idx[d]*dstRegion[d].Step) * dstStrides[d]
		}
		copy(out[dstEl*int64(elemSize):(dstEl+1)*int64(elemSize)], src[srcEl*int64(elemSize):(srcEl+1)*int64(elemSize)])
		for d := rank - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < regionLen[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out
}

func elemCount(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

// fillBytes returns a deterministic, non-uniform byte buffer of n bytes so
// a wrong-offset bug reliably produces a mismatch (all-zero buffers would
// hide many bugs).
func fillBytes(n int64) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*7 + 13)
	}
	return b
}

// TestCopyMatchesNaive_Basic covers a handful of hand-picked cases spanning
// every original call site's shape (whole chunk, edge-clipped chunk,
// sliced/strided selection, sharding inner<->outer regions, rank-0).
func TestCopyMatchesNaive_Basic(t *testing.T) {
	const esz = 4
	cases := []struct {
		name                 string
		srcShape, dstShape   []int64
		srcRegion, dstRegion []Region
	}{
		{
			name:     "whole_chunk_2d",
			srcShape: []int64{4, 4}, dstShape: []int64{4, 4},
			srcRegion: []Region{{0, 4, 1}, {0, 4, 1}},
			dstRegion: []Region{{0, 4, 1}, {0, 4, 1}},
		},
		{
			name:     "edge_clip_2d",
			srcShape: []int64{4, 4}, dstShape: []int64{5, 5},
			srcRegion: []Region{{0, 4, 1}, {0, 1, 1}},
			dstRegion: []Region{{0, 4, 1}, {4, 5, 1}},
		},
		{
			name:     "strided_1d",
			srcShape: []int64{10}, dstShape: []int64{5},
			srcRegion: []Region{{0, 10, 2}},
			dstRegion: []Region{{0, 5, 1}},
		},
		{
			name:     "strided_2d_mixed",
			srcShape: []int64{8, 8}, dstShape: []int64{4, 8},
			srcRegion: []Region{{0, 8, 2}, {0, 8, 1}},
			dstRegion: []Region{{0, 4, 1}, {0, 8, 1}},
		},
		{
			name:     "sharding_inner_into_outer",
			srcShape: []int64{2, 2}, dstShape: []int64{6, 6},
			srcRegion: []Region{{0, 2, 1}, {0, 2, 1}},
			dstRegion: []Region{{2, 4, 1}, {4, 6, 1}},
		},
		{
			name:     "rank0_scalar",
			srcShape: []int64{}, dstShape: []int64{},
			srcRegion: []Region{}, dstRegion: []Region{},
		},
		{
			name:     "rank3_partial",
			srcShape: []int64{4, 4, 4}, dstShape: []int64{2, 3, 4},
			srcRegion: []Region{{1, 3, 1}, {0, 3, 1}, {0, 4, 1}},
			dstRegion: []Region{{0, 2, 1}, {0, 3, 1}, {0, 4, 1}},
		},
		{
			name: "dst_stepped_write_direction",
			// Mirrors writeOneChunk: the DESTINATION side is stepped, not
			// the source (writing a strided selection into a chunk).
			srcShape: []int64{5}, dstShape: []int64{10},
			srcRegion: []Region{{0, 5, 1}},
			dstRegion: []Region{{0, 10, 2}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srcBuf := fillBytes(elemCount(tc.srcShape) * esz)
			dstBuf := fillBytes(elemCount(tc.dstShape) * esz)

			want := naiveCopy(dstBuf, tc.dstShape, tc.dstRegion, srcBuf, tc.srcShape, tc.srcRegion, esz)

			got := append([]byte(nil), dstBuf...)
			Copy(got, tc.dstShape, tc.dstRegion, srcBuf, tc.srcShape, tc.srcRegion, esz, nil)

			if !bytes.Equal(got, want) {
				t.Fatalf("Copy mismatch:\n got=%v\nwant=%v", got, want)
			}
		})
	}
}

// randomTrial generates a random valid (shapes, regions, elemSize)
// combination from r, runs Copy against naiveCopy, and reports a mismatch
// via t.Fatalf/t.Errorf (whichever the caller's *testing.T supports).
// Shared by TestCopyMatchesNaive_Random and FuzzCopy so both exercise
// exactly the same generation logic and failure format.
func randomTrial(t testing.TB, r *rand.Rand, sc *Scratch, label string) {
	rank := 1 + r.Intn(4)
	esz := []int{1, 2, 4, 8}[r.Intn(4)]

	srcShape := make([]int64, rank)
	dstShape := make([]int64, rank)
	srcRegion := make([]Region, rank)
	dstRegion := make([]Region, rank)
	for d := 0; d < rank; d++ {
		srcShape[d] = int64(1 + r.Intn(6))
		dstShape[d] = int64(1 + r.Intn(6))

		// Pick a source sub-range, optionally stepped.
		sStart := int64(r.Intn(int(srcShape[d])))
		sLen := int64(1 + r.Intn(int(srcShape[d]-sStart)))
		sStep := int64(1)
		if r.Intn(2) == 0 && sLen > 1 {
			sStep = int64(1 + r.Intn(2))
		}
		sStop := sStart + (sLen-1)*sStep + 1
		if sStop > srcShape[d] {
			sStop = srcShape[d]
		}
		srcRegion[d] = Region{Start: sStart, Stop: sStop, Step: sStep}
		n := srcRegion[d].Length()
		if n == 0 {
			n = 1
			srcRegion[d] = Region{Start: 0, Stop: 1, Step: 1}
			srcShape[d] = 1
		}

		// Destination range must have the same element count; keep it
		// step==1 or ==sStep at random, sized to match n exactly.
		dStep := int64(1)
		if r.Intn(2) == 0 {
			dStep = sStep
		}
		dSpan := (n-1)*dStep + 1
		maxStart := dstShape[d] - dSpan
		if maxStart < 0 {
			dstShape[d] = dSpan
			maxStart = 0
		}
		dStart := int64(0)
		if maxStart > 0 {
			dStart = int64(r.Intn(int(maxStart) + 1))
		}
		dstRegion[d] = Region{Start: dStart, Stop: dStart + dSpan, Step: dStep}
	}

	srcBuf := fillBytes(elemCount(srcShape) * int64(esz))
	dstBuf := fillBytes(elemCount(dstShape) * int64(esz))

	want := naiveCopy(dstBuf, dstShape, dstRegion, srcBuf, srcShape, srcRegion, esz)

	got := append([]byte(nil), dstBuf...)
	Copy(got, dstShape, dstRegion, srcBuf, srcShape, srcRegion, esz, sc)

	if !bytes.Equal(got, want) {
		t.Fatalf("%s: mismatch\nrank=%d esz=%d\nsrcShape=%v srcRegion=%v\ndstShape=%v dstRegion=%v\n got=%v\nwant=%v",
			label, rank, esz, srcShape, srcRegion, dstShape, dstRegion, got, want)
	}
}

// TestCopyMatchesNaive_Random fuzzes random shapes/regions/element sizes
// against the naive reference, exercising the contiguous-merge fast path
// across many rank/shape combinations including scratch reuse across calls.
func TestCopyMatchesNaive_Random(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	var sc Scratch
	for trial := 0; trial < 2000; trial++ {
		randomTrial(t, r, &sc, fmt.Sprintf("trial %d", trial))
	}
}

// FuzzCopy hands the corpus/fuzzer-provided bytes to randomTrial as a random
// seed, so every fuzz input deterministically selects one random valid
// (shape, region, elemSize) combination via the same generator
// TestCopyMatchesNaive_Random uses, then checks Copy against naiveCopy.
func FuzzCopy(f *testing.F) {
	f.Add(int64(42))
	f.Add(int64(0))
	f.Add(int64(1))
	f.Fuzz(func(t *testing.T, seed int64) {
		r := rand.New(rand.NewSource(seed))
		var sc Scratch
		randomTrial(t, r, &sc, fmt.Sprintf("seed %d", seed))
	})
}

// TestCopyZeroLength checks a zero-length region is a safe no-op.
func TestCopyZeroLength(t *testing.T) {
	dst := fillBytes(16)
	before := append([]byte(nil), dst...)
	src := fillBytes(16)
	Copy(dst, []int64{4}, []Region{{2, 2, 1}}, src, []int64{4}, []Region{{0, 0, 1}}, 4, nil)
	if !bytes.Equal(dst, before) {
		t.Fatalf("zero-length region mutated dst: got=%v want=%v", dst, before)
	}
}

// TestScratchReuseAcrossRanks checks a Scratch sized for a small rank is
// safely grown (not corrupted) when reused for a larger rank.
func TestScratchReuseAcrossRanks(t *testing.T) {
	sc := NewScratch(1)
	src1 := fillBytes(4 * 4)
	dst1 := make([]byte, 4*4)
	Copy(dst1, []int64{4}, []Region{{0, 4, 1}}, src1, []int64{4}, []Region{{0, 4, 1}}, 4, sc)
	if !bytes.Equal(dst1, src1) {
		t.Fatalf("rank-1 copy via small scratch failed: got=%v want=%v", dst1, src1)
	}

	src2 := fillBytes(3 * 3 * 3 * 4)
	dst2 := make([]byte, 3*3*3*4)
	region := []Region{{0, 3, 1}, {0, 3, 1}, {0, 3, 1}}
	Copy(dst2, []int64{3, 3, 3}, region, src2, []int64{3, 3, 3}, region, 4, sc)
	if !bytes.Equal(dst2, src2) {
		t.Fatalf("rank-3 copy via grown scratch failed: got=%v want=%v", dst2, src2)
	}
}
