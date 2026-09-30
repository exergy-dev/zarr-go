// Package ndcopy provides a single strided N-dimensional byte-buffer copy
// kernel shared by every part of the codebase that moves a rectangular
// sub-region between two row-major buffers: Array[T].Read/Write's chunk<->
// user-buffer copies, and the sharding codec's shard<->inner-chunk and
// partial-decode region copies. Consolidating these (they were previously
// five near-identical hand-rolled implementations) means a fix or
// optimization to the copy logic lands everywhere at once.
package ndcopy

import "github.com/exergy-dev/zarr-go/codec"

// Region is a half-open range [Start, Stop) along one dimension, with a
// positive Step. Step=1 selects every element in the range. It is the same
// type as codec.DimRange, so partial-decode regions and per-chunk
// selections feed Copy without conversion.
type Region = codec.DimRange

// length returns the number of elements r selects.
func length(r Region) int64 {
	if r.Step <= 0 || r.Stop <= r.Start {
		return 0
	}
	return (r.Stop - r.Start + r.Step - 1) / r.Step
}

// Scratch holds reusable buffers for Copy, so a hot loop calling Copy many
// times (e.g. once per chunk) can avoid a per-call allocation by reusing one
// Scratch sized for the max rank it will see. A nil Scratch makes Copy
// allocate its own (fine for cold/one-off calls).
type Scratch struct {
	SrcStrides, DstStrides []int64
	RegionLen, Idx         []int64
}

// NewScratch allocates a Scratch sized for the given rank.
func NewScratch(rank int) *Scratch {
	return &Scratch{
		SrcStrides: make([]int64, rank),
		DstStrides: make([]int64, rank),
		RegionLen:  make([]int64, rank),
		Idx:        make([]int64, rank),
	}
}

// slices returns rank-length slices of each Scratch buffer, growing/
// allocating them first if its buffers are too short (so a Scratch sized
// for a smaller rank can still be reused safely). A nil Scratch gets all
// four carved from a single allocation.
func (sc *Scratch) slices(rank int) (srcStrides, dstStrides, regionLen, idx []int64) {
	if sc == nil {
		buf := make([]int64, 4*rank)
		return buf[:rank:rank], buf[rank : 2*rank : 2*rank], buf[2*rank : 3*rank : 3*rank], buf[3*rank:]
	}
	if len(sc.SrcStrides) < rank {
		sc.SrcStrides = make([]int64, rank)
	}
	if len(sc.DstStrides) < rank {
		sc.DstStrides = make([]int64, rank)
	}
	if len(sc.RegionLen) < rank {
		sc.RegionLen = make([]int64, rank)
	}
	if len(sc.Idx) < rank {
		sc.Idx = make([]int64, rank)
	}
	return sc.SrcStrides[:rank], sc.DstStrides[:rank], sc.RegionLen[:rank], sc.Idx[:rank]
}

// rowMajorStridesInto fills out (len(shape)) with the row-major element
// strides for shape.
func rowMajorStridesInto(shape, out []int64) {
	rank := len(shape)
	if rank == 0 {
		return
	}
	out[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		out[i] = out[i+1] * shape[i+1]
	}
}

// Copy copies elemSize-byte elements from the hyperrectangle srcRegion of
// src (a row-major buffer of shape srcShape) into the hyperrectangle
// dstRegion of dst (a row-major buffer of shape dstShape). srcRegion[d] and
// dstRegion[d] must have equal Length() for every dimension d — Copy does
// not validate this.
//
// Fast path: when every Region in both srcRegion and dstRegion has Step==1,
// contiguous trailing dimensions collapse into single memmove (copy) calls
// per outer-iteration position, rather than one copy per element. This is
// the dominant case (whole or sliced chunk reads/writes); a stepped
// selection along any dimension falls back to one copy per element.
//
// sc may be nil, in which case Copy makes one small allocation for its own
// scratch space; a loop calling Copy many times on one goroutine can pass
// a reused *Scratch to avoid even that.
func Copy(
	dst []byte, dstShape []int64, dstRegion []Region,
	src []byte, srcShape []int64, srcRegion []Region,
	elemSize int,
	sc *Scratch,
) {
	rank := len(srcShape)
	if rank == 0 {
		// Scalar: a single element, no dims to index — both buffers are
		// exactly one element and the copy is unconditionally at offset 0.
		copy(dst[:elemSize], src[:elemSize])
		return
	}

	srcStrides, dstStrides, regionLen, idx := sc.slices(rank)
	rowMajorStridesInto(srcShape, srcStrides)
	rowMajorStridesInto(dstShape, dstStrides)
	for i := 0; i < rank; i++ {
		regionLen[i] = length(dstRegion[i])
		idx[i] = 0
	}
	total := int64(1)
	for _, n := range regionLen {
		total *= n
	}
	if total == 0 {
		return
	}

	allStep1 := true
	for d := 0; d < rank; d++ {
		if srcRegion[d].Step != 1 || dstRegion[d].Step != 1 {
			allStep1 = false
			break
		}
	}

	if allStep1 {
		copyContiguous(dst, dstShape, dstRegion, dstStrides, src, srcShape, srcRegion, srcStrides, regionLen, idx, elemSize)
		return
	}

	// Fallback: stride > 1 in some dimension on at least one side —
	// element-by-element copy. idx was zeroed above.
	for k := int64(0); k < total; k++ {
		var srcEl, dstEl int64
		for d := 0; d < rank; d++ {
			srcEl += (srcRegion[d].Start + idx[d]*srcRegion[d].Step) * srcStrides[d]
			dstEl += (dstRegion[d].Start + idx[d]*dstRegion[d].Step) * dstStrides[d]
		}
		copy(
			dst[dstEl*int64(elemSize):(dstEl+1)*int64(elemSize)],
			src[srcEl*int64(elemSize):(srcEl+1)*int64(elemSize)],
		)
		for d := rank - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < regionLen[d] {
				break
			}
			idx[d] = 0
		}
	}
}

// copyContiguous handles the all-Step==1 case: it collapses the maximal run
// of trailing dimensions that span the FULL extent on both the source dim
// (srcShape) and destination dim (dstShape) into a single memmove per
// outer-iteration position (the outermost merged dimension itself may be a
// partial range — only the dims strictly inside it need full span).
func copyContiguous(
	dst []byte, dstShape []int64, dstRegion []Region, dstStrides []int64,
	src []byte, srcShape []int64, srcRegion []Region, srcStrides []int64,
	regionLen, idx []int64,
	elemSize int,
) {
	rank := len(srcShape)
	mergeStart := rank - 1
	for mergeStart > 0 {
		d := mergeStart
		if srcRegion[d].Start != 0 || srcRegion[d].Stop != srcShape[d] ||
			dstRegion[d].Start != 0 || dstRegion[d].Stop != dstShape[d] {
			break
		}
		mergeStart--
	}

	innerElems := int64(1)
	for d := mergeStart; d < rank; d++ {
		innerElems *= regionLen[d]
	}
	innerBytes := innerElems * int64(elemSize)

	outerRank := mergeStart
	if outerRank == 0 {
		var srcEl, dstEl int64
		for d := 0; d < rank; d++ {
			srcEl += srcRegion[d].Start * srcStrides[d]
			dstEl += dstRegion[d].Start * dstStrides[d]
		}
		srcOff := srcEl * int64(elemSize)
		dstOff := dstEl * int64(elemSize)
		copy(dst[dstOff:dstOff+innerBytes], src[srcOff:srcOff+innerBytes])
		return
	}

	var innerSrcBase, innerDstBase int64
	for d := mergeStart; d < rank; d++ {
		innerSrcBase += srcRegion[d].Start * srcStrides[d]
		innerDstBase += dstRegion[d].Start * dstStrides[d]
	}

	outerIdx := idx[:outerRank]
	for d := 0; d < outerRank; d++ {
		outerIdx[d] = 0
	}
	outerTotal := int64(1)
	for d := 0; d < outerRank; d++ {
		outerTotal *= regionLen[d]
	}
	for k := int64(0); k < outerTotal; k++ {
		var srcEl, dstEl int64
		for d := 0; d < outerRank; d++ {
			srcEl += (srcRegion[d].Start + outerIdx[d]) * srcStrides[d]
			dstEl += (dstRegion[d].Start + outerIdx[d]) * dstStrides[d]
		}
		srcOff := (srcEl + innerSrcBase) * int64(elemSize)
		dstOff := (dstEl + innerDstBase) * int64(elemSize)
		copy(dst[dstOff:dstOff+innerBytes], src[srcOff:srcOff+innerBytes])
		for d := outerRank - 1; d >= 0; d-- {
			outerIdx[d]++
			if outerIdx[d] < regionLen[d] {
				break
			}
			outerIdx[d] = 0
		}
	}
}
