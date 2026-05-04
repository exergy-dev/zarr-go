package zarr

import (
	"fmt"
	"iter"
)

// Selection identifies a contiguous (with stride) hyperrectangle of an array.
// One DimRange per dimension; len(Ranges) must equal the array rank.
type Selection struct {
	Ranges []DimRange
}

// DimRange is the [Start, Stop) range with Step along one dimension.
// Step must be positive. Step=1 selects all elements in the range.
type DimRange struct {
	Start, Stop, Step int64
}

// All returns a Selection that covers the entire array of the given shape.
func All(shape []int64) Selection {
	rs := make([]DimRange, len(shape))
	for i, n := range shape {
		rs[i] = DimRange{0, n, 1}
	}
	return Selection{Ranges: rs}
}

// NumElements returns the count of elements covered by the selection.
func (s Selection) NumElements() int64 {
	if len(s.Ranges) == 0 {
		return 0
	}
	n := int64(1)
	for _, r := range s.Ranges {
		n *= r.Length()
	}
	return n
}

// Length returns the number of elements selected along this dim.
func (r DimRange) Length() int64 {
	if r.Step <= 0 || r.Stop <= r.Start {
		return 0
	}
	return (r.Stop - r.Start + r.Step - 1) / r.Step
}

// Validate checks the selection against the given shape.
func (s Selection) Validate(shape []int64) error {
	if len(s.Ranges) != len(shape) {
		return fmt.Errorf("selection rank %d != array rank %d", len(s.Ranges), len(shape))
	}
	for i, r := range s.Ranges {
		if r.Step <= 0 {
			return fmt.Errorf("dim %d: step must be positive, got %d", i, r.Step)
		}
		if r.Start < 0 || r.Stop > shape[i] || r.Start > r.Stop {
			return fmt.Errorf("dim %d: range [%d,%d) out of bounds for shape %d", i, r.Start, r.Stop, shape[i])
		}
	}
	return nil
}

// ChunkCoord identifies one chunk in the chunk grid.
type ChunkCoord []int64

// Clone returns a copy of c so the caller can retain it across iterator steps.
func (c ChunkCoord) Clone() ChunkCoord {
	out := make(ChunkCoord, len(c))
	copy(out, c)
	return out
}

// ChunksIntersecting yields each chunk coordinate whose chunk overlaps the
// selection. Coordinates are emitted in row-major (C) order.
//
// The yielded slice is reused across iterations; clone it if retained.
func ChunksIntersecting(arrayShape, chunkShape []int64, sel Selection) iter.Seq[ChunkCoord] {
	return func(yield func(ChunkCoord) bool) {
		rank := len(arrayShape)
		if rank == 0 || len(chunkShape) != rank || len(sel.Ranges) != rank {
			return
		}
		// Per-dim chunk index range: chunks that touch [Start, Stop).
		lo := make([]int64, rank)
		hi := make([]int64, rank)
		for i := 0; i < rank; i++ {
			r := sel.Ranges[i]
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

// ChunkSelection returns the per-dim ranges of (a) the source region inside the
// chunk and (b) the destination region inside the selection's output, both for
// the chunk at coord. Both ranges have the same length per dim.
//
// chunkSrc[i].Start..Stop is in chunk-local coordinates with Step=Selection.Step.
// outDst[i].Start..Stop is in output-element coordinates with Step=1.
func ChunkSelection(arrayShape, chunkShape []int64, sel Selection, coord ChunkCoord) (chunkSrc, outDst []DimRange) {
	rank := len(arrayShape)
	chunkSrc = make([]DimRange, rank)
	outDst = make([]DimRange, rank)
	chunkSelectionInto(arrayShape, chunkShape, sel, coord, chunkSrc, outDst)
	return chunkSrc, outDst
}

// chunkSelectionInto fills caller-supplied chunkSrc and outDst slices (each of
// len rank) with the chunk-local source range and output destination range for
// the chunk at coord. This avoids allocating new slices per chunk.
func chunkSelectionInto(arrayShape, chunkShape []int64, sel Selection, coord ChunkCoord, chunkSrc, outDst []DimRange) {
	rank := len(arrayShape)
	for i := 0; i < rank; i++ {
		r := sel.Ranges[i]
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
		chunkSrc[i] = DimRange{Start: first - chunkOrigin, Stop: stop - chunkOrigin, Step: r.Step}
		outStart := (first - r.Start) / r.Step
		outStop := outStart + chunkSrc[i].Length()
		outDst[i] = DimRange{Start: outStart, Stop: outStop, Step: 1}
	}
}
