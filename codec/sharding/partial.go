// Partial decode for sharding_indexed.
//
// DecodePartial implements codec.PartialDecoder. Given a sub-region of the
// outer shard (in chunk-local coordinates) and a RangeFetcher pointing at the
// shard blob, it:
//
//  1. Range-fetches the index trailer (small, few KB).
//  2. Decodes the index to recover (offset, length) for every inner chunk.
//  3. Identifies the inner chunks intersecting the requested region.
//  4. Coalesces their byte ranges (using internal/scheduler.CoalesceRanges)
//     and issues one range read per coalesced span.
//  5. Decodes only those inner chunks via the configured inner pipeline.
//  6. Copies their intersecting sub-regions into outBuf.
//
// outBuf must be pre-sized to product(region lengths) * element_size(dtype),
// and is written in row-major order matching the region's dense shape.

package sharding

import (
	"context"
	"fmt"

	"github.com/rkm/zarr-go/codec"
	"github.com/rkm/zarr-go/internal/scheduler"
)

// coalesceGapBytes is the maximum gap between two inner chunk byte ranges
// that will still be merged into a single fetch.
//
// Inner chunks are stored contiguously in the shard body in row-major grid
// order. For a 2D rectangular sub-region the intersecting inner chunks are
// usually NOT contiguous in that order — between two needed rows of inner
// chunks lie one or more unneeded inner chunks. The cost trade-off: paying
// for a few KB of extra payload is cheaper than paying for the latency of
// another round trip on a high-RTT object store. 4 MiB is a generous cap
// that comfortably covers most "skip past an unneeded chunk row" gaps while
// still bounding the worst-case wasted bandwidth.
const coalesceGapBytes int64 = 4 * 1024 * 1024

// DecodePartial implements codec.PartialDecoder.
func (c *Codec) DecodePartial(
	ctx context.Context,
	fetcher codec.RangeFetcher,
	key string,
	fullShape []int64,
	dtype string,
	region []codec.DimRange,
	outBuf []byte,
) error {
	if c.inner == nil || c.index == nil {
		return fmt.Errorf("sharding_indexed: codec not configured")
	}
	if c.IndexLocation != IndexEnd {
		return fmt.Errorf("sharding_indexed: index_location=start not supported")
	}
	if len(region) != len(fullShape) {
		return fmt.Errorf("sharding_indexed: region rank %d != chunk rank %d", len(region), len(fullShape))
	}
	for i, r := range region {
		if r.Step != 1 {
			return fmt.Errorf("sharding_indexed: DecodePartial requires step==1, dim %d step=%d", i, r.Step)
		}
		if r.Start < 0 || r.Stop > fullShape[i] || r.Start > r.Stop {
			return fmt.Errorf("sharding_indexed: region dim %d [%d,%d) out of bounds for shape %d",
				i, r.Start, r.Stop, fullShape[i])
		}
	}
	esz := elemSize(dtype)
	if esz == 0 {
		return fmt.Errorf("sharding_indexed: unsupported dtype %q", dtype)
	}
	rank := len(fullShape)
	regionShape := make([]int64, rank)
	for i, r := range region {
		regionShape[i] = r.Stop - r.Start
	}
	expectedOut := elementCount(regionShape) * int64(esz)
	if int64(len(outBuf)) != expectedOut {
		return fmt.Errorf("sharding_indexed: outBuf size %d != expected %d", len(outBuf), expectedOut)
	}
	// Zero outBuf so that absent inner chunks (sentinel) read as fill (zero).
	for i := range outBuf {
		outBuf[i] = 0
	}
	if elementCount(regionShape) == 0 {
		return nil
	}

	grid, err := gridDims(fullShape, c.ChunkShape)
	if err != nil {
		return err
	}
	n := numInner(grid)

	// 1) Fetch the index trailer.
	//
	// The trailer length is a deterministic function of the inner-chunk
	// count and the index pipeline (computed by indexEncodedSize). To read
	// it we need the trailer's offset = shardLen - idxLen. The
	// codec.RangeFetcher interface doesn't expose blob length, so we prefer
	// SuffixRangeFetcher (cloud stores can serve `Range: bytes=-N` natively
	// and the engine adapter exposes it for local stores via cached length).
	// If no suffix API is available we fall back to fetching the whole shard
	// in one GetRange (large-N, clamped by the backend). That loses the
	// bandwidth win but preserves correctness and the memory/CPU wins.
	idxLen, err := c.indexEncodedSize(n)
	if err != nil {
		return fmt.Errorf("sharding_indexed: compute index size: %w", err)
	}

	var indexBytes []byte
	if sf, ok := fetcher.(SuffixRangeFetcher); ok {
		indexBytes, err = sf.GetSuffix(ctx, key, idxLen)
		if err != nil {
			return fmt.Errorf("sharding_indexed: fetch index trailer: %w", err)
		}
		if int64(len(indexBytes)) != idxLen {
			return fmt.Errorf("sharding_indexed: index trailer short: got %d want %d", len(indexBytes), idxLen)
		}
	} else {
		whole, err := fetcher.GetRange(ctx, key, 0, maxRangeBytes)
		if err != nil {
			return fmt.Errorf("sharding_indexed: fetch shard: %w", err)
		}
		shardLen := int64(len(whole))
		if shardLen < idxLen {
			return fmt.Errorf("sharding_indexed: shard too short: %d < %d", shardLen, idxLen)
		}
		indexBytes = whole[shardLen-idxLen:]
		fetcher = &slicedFetcher{key: key, base: whole}
	}

	entries, err := c.decodeIndex(indexBytes, n)
	if err != nil {
		return err
	}

	// 2) Identify inner chunks intersecting the region.
	// For each inner chunk grid coord, compute its element-space extent in
	// chunk-local coordinates and intersect with region.
	type needed struct {
		coord      []int64
		off        int64
		ln         int64
		innerShape []int64
		// localRegion is the inner-chunk-local range to extract.
		localRegion []innerRange
	}
	var needs []needed

	coord := make([]int64, rank)
	for i := int64(0); i < n; i++ {
		off := entries[2*i]
		ln := entries[2*i+1]
		// Compute this inner chunk's element extents in chunk-local space.
		var origin = make([]int64, rank)
		var end = make([]int64, rank)
		intersects := true
		for d := 0; d < rank; d++ {
			origin[d] = coord[d] * c.ChunkShape[d]
			end[d] = origin[d] + c.ChunkShape[d]
			if end[d] > fullShape[d] {
				end[d] = fullShape[d]
			}
			if origin[d] >= region[d].Stop || end[d] <= region[d].Start {
				intersects = false
			}
		}
		if intersects {
			if off == emptyMarker && ln == emptyMarker {
				// All-zero inner chunk: outBuf is already zeroed; skip.
			} else {
				innerShape := innerShapeAt(coord, c.ChunkShape, fullShape)
				lr := make([]innerRange, rank)
				for d := 0; d < rank; d++ {
					lo := region[d].Start - origin[d]
					if lo < 0 {
						lo = 0
					}
					hi := region[d].Stop - origin[d]
					if hi > end[d]-origin[d] {
						hi = end[d] - origin[d]
					}
					lr[d] = innerRange{
						innerStart: lo,
						innerStop:  hi,
						outStart:   origin[d] + lo - region[d].Start,
					}
				}
				needs = append(needs, needed{
					coord:       append([]int64(nil), coord...),
					off:         int64(off),
					ln:          int64(ln),
					innerShape:  innerShape,
					localRegion: lr,
				})
			}
		}
		if !nextCoord(coord, grid) && i != n-1 {
			return fmt.Errorf("sharding_indexed: ran out of grid coords")
		}
	}

	if len(needs) == 0 {
		return nil
	}

	// 3) Coalesce ranges and fetch.
	ranges := make([]scheduler.Range, len(needs))
	for i, nd := range needs {
		ranges[i] = scheduler.Range{Off: nd.off, N: nd.ln}
	}
	merged := scheduler.CoalesceRanges(ranges, coalesceGapBytes)

	// Pre-allocate per-need raw bytes slot.
	rawByIdx := make([][]byte, len(needs))
	for _, m := range merged {
		blob, err := fetcher.GetRange(ctx, key, m.Off, m.N)
		if err != nil {
			return fmt.Errorf("sharding_indexed: fetch inner range [%d,%d): %w", m.Off, m.Off+m.N, err)
		}
		if int64(len(blob)) < m.N {
			return fmt.Errorf("sharding_indexed: short read [%d,%d): got %d", m.Off, m.Off+m.N, len(blob))
		}
		for _, mem := range m.Members {
			start := mem.OffInMerged
			rawByIdx[mem.OriginalIndex] = blob[start : start+mem.N]
		}
	}

	// 4) Decode each needed inner chunk and copy its intersection into outBuf.
	for i, nd := range needs {
		raw := rawByIdx[i]
		innerChunk := codec.ArrayChunk{Shape: nd.innerShape, Dtype: dtype}
		if err := c.inner.Decode(raw, &innerChunk); err != nil {
			return fmt.Errorf("sharding_indexed: decode inner chunk %v: %w", nd.coord, err)
		}
		copySubregion(
			outBuf, regionShape,
			innerChunk.Data, nd.innerShape,
			nd.localRegion, esz,
		)
	}
	return nil
}

// maxRangeBytes is used as the "fetch everything available" sentinel for
// fallback range reads when the fetcher doesn't expose a suffix API. All
// in-tree store backends clamp n to the remaining blob length.
const maxRangeBytes int64 = 1 << 62

// SuffixRangeFetcher is an optional extension implemented by stores that can
// natively serve a "last N bytes" request without knowing the blob length up
// front. HTTP and S3-style object stores can serve this via the
// `Range: bytes=-N` header. Local stores typically can; mem currently can't,
// so this codepath is taken when wired by the engine through a shim.
type SuffixRangeFetcher interface {
	GetSuffix(ctx context.Context, key string, n int64) ([]byte, error)
}

// slicedFetcher serves range reads against an already-fetched whole blob.
// It's used as a fallback when we had to fetch the whole shard to find the
// trailer — subsequent inner reads are then satisfied from memory rather
// than re-issued to the network.
type slicedFetcher struct {
	key  string
	base []byte
}

func (s *slicedFetcher) GetRange(_ context.Context, key string, off, n int64) ([]byte, error) {
	if key != s.key {
		return nil, fmt.Errorf("sharding_indexed: unexpected key %q (want %q)", key, s.key)
	}
	if off < 0 || off > int64(len(s.base)) {
		return nil, fmt.Errorf("sharding_indexed: offset %d out of range", off)
	}
	end := off + n
	if end > int64(len(s.base)) {
		end = int64(len(s.base))
	}
	return s.base[off:end], nil
}

// innerRange describes the inner-chunk-local sub-range we want to extract from
// one inner chunk, along with where it lands in the region-shaped output.
type innerRange struct {
	innerStart, innerStop int64 // in inner-chunk local coords
	outStart              int64 // in region-shaped output coords
}

// copySubregion copies the inner-chunk-local rectangle described by lr from
// inner (shape innerShape) into outBuf (shape regionShape). All buffers are
// row-major; elements are esz bytes.
func copySubregion(
	outBuf []byte, regionShape []int64,
	inner []byte, innerShape []int64,
	lr []innerRange, esz int,
) {
	rank := len(regionShape)
	innerStrides := make([]int64, rank)
	outStrides := make([]int64, rank)
	innerStrides[rank-1] = 1
	outStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		innerStrides[i] = innerStrides[i+1] * innerShape[i+1]
		outStrides[i] = outStrides[i+1] * regionShape[i+1]
	}
	rowElems := lr[rank-1].innerStop - lr[rank-1].innerStart
	if rowElems <= 0 {
		return
	}
	rowBytes := rowElems * int64(esz)
	innerRowBase := lr[rank-1].innerStart
	outRowBase := lr[rank-1].outStart

	if rank == 1 {
		copy(
			outBuf[outRowBase*int64(esz):outRowBase*int64(esz)+rowBytes],
			inner[innerRowBase*int64(esz):innerRowBase*int64(esz)+rowBytes],
		)
		return
	}

	// Iterate over the leading rank-1 dims.
	pos := make([]int64, rank-1)
	for d := 0; d < rank-1; d++ {
		pos[d] = 0
	}
	regionLens := make([]int64, rank-1)
	for d := 0; d < rank-1; d++ {
		regionLens[d] = lr[d].innerStop - lr[d].innerStart
		if regionLens[d] <= 0 {
			return
		}
	}
	for {
		var innerElemOff, outElemOff int64
		for d := 0; d < rank-1; d++ {
			innerElemOff += (lr[d].innerStart + pos[d]) * innerStrides[d]
			outElemOff += (lr[d].outStart + pos[d]) * outStrides[d]
		}
		innerElemOff += innerRowBase
		outElemOff += outRowBase
		copy(
			outBuf[outElemOff*int64(esz):outElemOff*int64(esz)+rowBytes],
			inner[innerElemOff*int64(esz):innerElemOff*int64(esz)+rowBytes],
		)
		// Advance pos in row-major (last leading dim fastest).
		i := rank - 2
		for i >= 0 {
			pos[i]++
			if pos[i] < regionLens[i] {
				break
			}
			pos[i] = 0
			i--
		}
		if i < 0 {
			return
		}
	}
}
