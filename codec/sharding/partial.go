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
//  6. Copies their intersecting sub-regions into req.Out.
//
// req.Out must be pre-sized to product(region lengths) * element_size(dtype),
// and is written in row-major order matching the region's dense shape.

package sharding

import (
	"context"
	"fmt"
	"math"

	"github.com/rkm/zarr-go/codec"
	"github.com/rkm/zarr-go/internal/ndcopy"
	"github.com/rkm/zarr-go/internal/scheduler"
)

// DecodePartial implements codec.PartialDecoder.
//
// req.Fill, when non-nil, is one element's worth of native-LE bytes
// representing the array's fill value; req.Out is pre-filled with it before
// decoding so any region overlapping an absent inner chunk reads back as
// fill.
func (c *Codec) DecodePartial(ctx context.Context, fetcher codec.RangeFetcher, req *codec.PartialReadRequest) error {
	if c.inner == nil || c.index == nil {
		return fmt.Errorf("sharding_indexed: codec not configured")
	}
	if c.IndexLocation != IndexEnd && c.IndexLocation != IndexStart {
		return fmt.Errorf("sharding_indexed: unknown index_location %d", c.IndexLocation)
	}
	fullShape := req.FullShape
	region := req.Region
	dtype := req.Dtype
	fill := req.Fill
	outBuf := req.Out
	gapBytes := req.GapBytes
	if gapBytes < 0 {
		gapBytes = 0
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
	esz := codec.DtypeSize(dtype)
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
	// Pre-fill outBuf with fill so any region overlapping an absent inner
	// chunk reads as the array's fill_value (not zero). Empty/nil fill
	// degenerates to zero-fill via writeFill.
	if len(fill) != esz {
		fill = nil
	}
	writeFill(outBuf, fill)
	if elementCount(regionShape) == 0 {
		return nil
	}

	grid, err := gridDims(fullShape, c.ChunkShape)
	if err != nil {
		return err
	}
	n := numInner(grid)
	key := req.Key

	// 1) Fetch the index payload at the configured location.
	idxLen, err := c.indexEncodedSize(n)
	if err != nil {
		return fmt.Errorf("sharding_indexed: compute index size: %w", err)
	}

	var indexBytes []byte
	if c.IndexLocation == IndexEnd {
		if sf, ok := fetcher.(codec.SuffixRangeFetcher); ok {
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
	} else { // IndexStart
		// The index sits at offset 0; one positional range read is enough.
		head, err := fetcher.GetRange(ctx, key, 0, idxLen)
		if err != nil {
			return fmt.Errorf("sharding_indexed: fetch index header: %w", err)
		}
		if int64(len(head)) != idxLen {
			return fmt.Errorf("sharding_indexed: index header short: got %d want %d", len(head), idxLen)
		}
		indexBytes = head
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
				// off/ln come from untrusted stored metadata; reject values
				// that don't fit in an int64 before ever casting them (a
				// silent uint64->int64 wrap could otherwise produce a
				// negative offset/length downstream).
				if off > math.MaxInt64 || ln > math.MaxInt64 {
					return fmt.Errorf("sharding_indexed: index entry %d out of range (off=%d len=%d)", i, off, ln)
				}
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
	merged := scheduler.CoalesceRanges(ranges, gapBytes)

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
		innerChunk := codec.ArrayChunk{Shape: nd.innerShape, Dtype: dtype, Fill: fill}
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
// inner (shape innerShape) into outBuf (shape regionShape), via the shared
// ndcopy kernel. All buffers are row-major; elements are esz bytes.
func copySubregion(
	outBuf []byte, regionShape []int64,
	inner []byte, innerShape []int64,
	lr []innerRange, esz int,
) {
	rank := len(regionShape)
	srcRegion := make([]ndcopy.Region, rank)
	dstRegion := make([]ndcopy.Region, rank)
	for d := 0; d < rank; d++ {
		srcRegion[d] = ndcopy.Region{Start: lr[d].innerStart, Stop: lr[d].innerStop, Step: 1}
		dstRegion[d] = ndcopy.Region{Start: lr[d].outStart, Stop: lr[d].outStart + (lr[d].innerStop - lr[d].innerStart), Step: 1}
	}
	ndcopy.Copy(outBuf, regionShape, dstRegion, inner, innerShape, srcRegion, esz, nil)
}
