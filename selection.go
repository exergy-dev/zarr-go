package zarr

import (
	"fmt"
	"iter"
	"slices"

	"github.com/exergy-dev/zarr-go/codec"
	"github.com/exergy-dev/zarr-go/internal/ndcopy"
)

// Selection identifies a strided hyperrectangle of an array: one DimRange
// per dimension, so len(sel) must equal the array rank. A zero Step selects
// every element in [Start, Stop), as does Step 1:
//
//	sel := zarr.Selection{{Start: 0, Stop: 1024}, {Start: 0, Stop: 1024}}
//
// All returns the selection covering a whole array.
type Selection []DimRange

// DimRange is the half-open range [Start, Stop) along one dimension, with a
// non-negative Step. Step 0 is equivalent to Step 1 (every element).
type DimRange = codec.DimRange

// All returns a Selection that covers the entire array of the given shape.
func All(shape []int64) Selection {
	sel := make(Selection, len(shape))
	for i, n := range shape {
		sel[i] = DimRange{Start: 0, Stop: n, Step: 1}
	}
	return sel
}

// NumElements returns the count of elements covered by the selection. A
// rank-0 selection (nil or empty — only valid for a rank-0 array; Validate
// enforces len(sel) == array rank) covers exactly the one scalar element.
func (s Selection) NumElements() int64 {
	n := int64(1)
	for _, r := range s {
		n *= r.Length()
	}
	return n
}

// Validate checks the selection against the given shape.
func (s Selection) Validate(shape []int64) error {
	if len(s) != len(shape) {
		return fmt.Errorf("selection rank %d != array rank %d", len(s), len(shape))
	}
	for i, r := range s {
		if r.Step < 0 {
			return fmt.Errorf("dim %d: step must be non-negative, got %d", i, r.Step)
		}
		if r.Start < 0 || r.Stop > shape[i] || r.Start > r.Stop {
			return fmt.Errorf("dim %d: range [%d,%d) out of bounds for shape %d", i, r.Start, r.Stop, shape[i])
		}
	}
	return nil
}

// normalized returns a copy of s with every zero Step replaced by 1, so the
// engine's arithmetic never sees a zero step. Validate must have passed.
func (s Selection) normalized() Selection {
	out := slices.Clone(s)
	for i := range out {
		if out[i].Step == 0 {
			out[i].Step = 1
		}
	}
	return out
}

// ChunkCoord identifies one chunk in the chunk grid.
type ChunkCoord []int64

// Clone returns a copy of c so the caller can retain it across iterator steps.
func (c ChunkCoord) Clone() ChunkCoord { return slices.Clone(c) }

// ChunksIntersecting yields each chunk coordinate whose chunk overlaps the
// selection. Coordinates are emitted in row-major (C) order.
//
// The yielded slice is reused across iterations; clone it if retained.
func ChunksIntersecting(arrayShape, chunkShape []int64, sel Selection) iter.Seq[ChunkCoord] {
	return func(yield func(ChunkCoord) bool) {
		rank := len(arrayShape)
		if len(chunkShape) != rank || len(sel) != rank {
			return
		}
		// Per-dim chunk index range: chunks that touch [Start, Stop).
		lo := make([]int64, rank)
		hi := make([]int64, rank)
		for i := 0; i < rank; i++ {
			r := sel[i]
			if r.Length() == 0 {
				return
			}
			lo[i] = r.Start / chunkShape[i]
			hi[i] = (r.Stop - 1) / chunkShape[i]
		}
		coord := make(ChunkCoord, rank)
		copy(coord, lo)
		for {
			if !yield(coord) {
				return
			}
			// Increment in row-major order (last dim fastest).
			d := rank - 1
			for d >= 0 {
				coord[d]++
				if coord[d] <= hi[d] {
					break
				}
				coord[d] = lo[d]
				d--
			}
			if d < 0 {
				return
			}
		}
	}
}

// chunkSelection returns, for the chunk at coord, (a) the source region
// inside the chunk and (b) the destination region inside the selection's
// output. Both have the same length per dim: chunkSrc is in chunk-local
// coordinates with the selection's Step, outDst in output-element
// coordinates with Step=1. They are ndcopy.Regions, ready for ndcopy.Copy,
// and share one allocation.
func chunkSelection(arrayShape, chunkShape []int64, sel Selection, coord ChunkCoord) (chunkSrc, outDst []ndcopy.Region) {
	rank := len(arrayShape)
	buf := make([]ndcopy.Region, 2*rank)
	chunkSrc, outDst = buf[:rank:rank], buf[rank:]
	for i := 0; i < rank; i++ {
		r := sel[i]
		chunkOrigin := coord[i] * chunkShape[i]
		chunkEnd := chunkOrigin + chunkShape[i]
		if chunkEnd > arrayShape[i] {
			chunkEnd = arrayShape[i]
		}
		// First selected element >= chunkOrigin.
		first := r.Start
		if first < chunkOrigin {
			delta := chunkOrigin - first
			steps := (delta + r.Step - 1) / r.Step
			first = first + steps*r.Step
		}
		// Last selected element < chunkEnd, also < r.Stop.
		stop := r.Stop
		if stop > chunkEnd {
			stop = chunkEnd
		}
		chunkSrc[i] = ndcopy.Region{Start: first - chunkOrigin, Stop: stop - chunkOrigin, Step: r.Step}
		outStart := (first - r.Start) / r.Step
		outStop := outStart + chunkSrc[i].Length()
		outDst[i] = ndcopy.Region{Start: outStart, Stop: outStop, Step: 1}
	}
	return chunkSrc, outDst
}
