// Package sharding implements the Zarr v3 "sharding_indexed" array→bytes codec.
//
// A shard is a single store blob containing many "inner chunks" plus a binary
// index trailer. Logically the shard is one chunk in the outer chunk grid; this
// codec subdivides it into inner chunks per the configured chunk_shape.
//
// Spec: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/v1.0.html
package sharding

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sync"

	"github.com/exergy-dev/zarr-go/codec"
	"github.com/exergy-dev/zarr-go/internal/declimit"
	"github.com/exergy-dev/zarr-go/internal/ndcopy"
)

// emptyMarker is the (offset, length) value used in the index for an inner
// chunk that is absent (logically all fill-value).
const emptyMarker uint64 = math.MaxUint64

// indexLocation indicates whether the index trailer is stored at the start or
// end of the shard.
type indexLocation uint8

const (
	// indexEnd places the index at the end of the shard (default).
	indexEnd indexLocation = iota
	// indexStart places the index at the start of the shard.
	indexStart
)

// Codec is the v3 "sharding_indexed" array→bytes codec. Construct it with
// New; the zero value is not usable. Its configuration is fixed at
// construction and it is safe for concurrent use.
type Codec struct {
	chunkShape    []int64
	indexLocation indexLocation

	inner *codec.Pipeline
	index *codec.Pipeline

	idxSizeMu    sync.Mutex
	idxSizeCache map[int64]int64 // n inner chunks -> encoded index trailer size
}

type configJSON struct {
	ChunkShape    []int64      `json:"chunk_shape"`
	Codecs        []codec.Spec `json:"codecs"`
	IndexCodecs   []codec.Spec `json:"index_codecs,omitempty"`
	IndexLocation *string      `json:"index_location,omitempty"`
}

// Name returns the v3 codec name.
func (*Codec) Name() string { return "sharding_indexed" }

// New builds a sharding_indexed codec from its v3 configuration object,
// including its inner and index pipelines (whose codecs must already be
// registered).
func New(raw json.RawMessage) (*Codec, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("sharding_indexed: missing configuration")
	}
	var cfg configJSON
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("sharding_indexed: invalid configuration: %w", err)
	}
	if len(cfg.ChunkShape) == 0 {
		return nil, fmt.Errorf("sharding_indexed: chunk_shape is required")
	}
	for i, d := range cfg.ChunkShape {
		if d <= 0 {
			return nil, fmt.Errorf("sharding_indexed: chunk_shape[%d]=%d must be positive", i, d)
		}
	}
	c := &Codec{
		chunkShape:    append([]int64(nil), cfg.ChunkShape...),
		indexLocation: indexEnd,
	}
	if cfg.IndexLocation != nil {
		switch *cfg.IndexLocation {
		case "end":
			c.indexLocation = indexEnd
		case "start":
			c.indexLocation = indexStart
		default:
			return nil, fmt.Errorf("sharding_indexed: invalid index_location %q", *cfg.IndexLocation)
		}
	}

	if len(cfg.Codecs) == 0 {
		return nil, fmt.Errorf("sharding_indexed: inner codecs is required")
	}
	inner, err := codec.BuildPipeline(cfg.Codecs)
	if err != nil {
		return nil, fmt.Errorf("sharding_indexed: build inner pipeline: %w", err)
	}
	c.inner = inner

	indexSpecs := cfg.IndexCodecs
	if len(indexSpecs) == 0 {
		indexSpecs = defaultIndexCodecs()
	}
	index, err := codec.BuildPipeline(indexSpecs)
	if err != nil {
		return nil, fmt.Errorf("sharding_indexed: build index pipeline: %w", err)
	}
	if len(index.AA) != 0 {
		// encodeIndex/decodeIndex only ever invoke index.AB and index.BB —
		// an array→array codec here would be silently skipped rather than
		// applied, which is worse than rejecting it up front.
		return nil, fmt.Errorf("sharding_indexed: array→array codecs are not supported in index_codecs")
	}
	c.index = index

	return c, nil
}

// defaultIndexCodecs is the default index_codecs pipeline:
// bytes (little) + crc32c.
func defaultIndexCodecs() []codec.Spec {
	return []codec.Spec{
		{Name: "bytes", Configuration: json.RawMessage(`{"endian":"little"}`)},
		{Name: "crc32c"},
	}
}

// elementCount returns product(shape). An empty shape (scalar) yields 1.
func elementCount(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

// gridDims computes the inner chunk grid: ceil(outerShape[i] / chunkShape[i]).
func gridDims(outerShape, chunkShape []int64) ([]int64, error) {
	if len(outerShape) != len(chunkShape) {
		return nil, fmt.Errorf("sharding_indexed: outer shape rank %d != chunk_shape rank %d",
			len(outerShape), len(chunkShape))
	}
	g := make([]int64, len(outerShape))
	for i := range outerShape {
		if outerShape[i] < 0 {
			return nil, fmt.Errorf("sharding_indexed: negative shard dim %d", outerShape[i])
		}
		g[i] = (outerShape[i] + chunkShape[i] - 1) / chunkShape[i]
		if g[i] == 0 {
			g[i] = 1
		}
	}
	return g, nil
}

// numInner returns product(grid).
func numInner(grid []int64) int64 {
	n := int64(1)
	for _, d := range grid {
		n *= d
	}
	return n
}

// indexEncodedSize returns the byte length of the encoded index trailer for n
// inner chunks. It encodes a dummy index of the right plain size to discover
// the trailer length added by the index pipeline, caching the result per n
// since a given Codec only ever sees a handful of distinct n (the full grid
// size, plus edge-shard grid sizes).
func (c *Codec) indexEncodedSize(n int64) (int64, error) {
	c.idxSizeMu.Lock()
	if v, ok := c.idxSizeCache[n]; ok {
		c.idxSizeMu.Unlock()
		return v, nil
	}
	c.idxSizeMu.Unlock()

	if n < 0 || n > math.MaxInt64/16 {
		return 0, fmt.Errorf("sharding_indexed: index entry count %d out of range", n)
	}
	plainSize := n * 16
	if err := declimit.CheckDeclaredSize(plainSize); err != nil {
		return 0, fmt.Errorf("sharding_indexed: index plain size: %w", err)
	}
	dummy := make([]byte, plainSize)
	chunk := codec.ArrayChunk{
		Shape: []int64{n * 2},
		Dtype: "uint64",
		Data:  dummy,
	}
	encoded, err := c.encodeIndex(chunk)
	if err != nil {
		return 0, err
	}
	size := int64(len(encoded))

	c.idxSizeMu.Lock()
	if c.idxSizeCache == nil {
		c.idxSizeCache = make(map[int64]int64)
	}
	c.idxSizeCache[n] = size
	c.idxSizeMu.Unlock()

	return size, nil
}

// encodeIndex runs the index pipeline (AB + BB) on an ArrayChunk view of the
// (offset, length) table.
func (c *Codec) encodeIndex(chunk codec.ArrayChunk) ([]byte, error) {
	buf, err := c.index.AB.EncodeAB(chunk)
	if err != nil {
		return nil, fmt.Errorf("encode index: %w", err)
	}
	for _, bb := range c.index.BB {
		buf, err = bb.EncodeBB(buf)
		if err != nil {
			return nil, fmt.Errorf("encode index (%s): %w", bb.Name(), err)
		}
	}
	return buf, nil
}

// decodeIndex runs the index pipeline in reverse to recover the raw
// (offset, length) table bytes for n inner chunks.
func (c *Codec) decodeIndex(in []byte, n int64) ([]uint64, error) {
	buf := in
	var err error
	for i := len(c.index.BB) - 1; i >= 0; i-- {
		buf, err = c.index.BB[i].DecodeBB(buf)
		if err != nil {
			return nil, fmt.Errorf("decode index (%s): %w", c.index.BB[i].Name(), err)
		}
	}
	out := codec.ArrayChunk{
		Shape: []int64{n * 2},
		Dtype: "uint64",
	}
	if err := c.index.AB.DecodeAB(buf, &out); err != nil {
		return nil, fmt.Errorf("decode index: %w", err)
	}
	if int64(len(out.Data)) != n*16 {
		return nil, fmt.Errorf("sharding_indexed: index size %d != expected %d", len(out.Data), n*16)
	}
	entries := make([]uint64, n*2)
	for i := range entries {
		entries[i] = binary.NativeEndian.Uint64(out.Data[i*8:])
	}
	return entries, nil
}

// encodeIndexEntries encodes a flat (offset, length) table into shard bytes.
func (c *Codec) encodeIndexEntries(entries []uint64) ([]byte, error) {
	buf := make([]byte, len(entries)*8)
	for i, v := range entries {
		binary.NativeEndian.PutUint64(buf[i*8:], v)
	}
	chunk := codec.ArrayChunk{
		Shape: []int64{int64(len(entries))},
		Dtype: "uint64",
		Data:  buf,
	}
	return c.encodeIndex(chunk)
}

// innerShapeAt returns the actual shape of the inner chunk at the given grid
// coordinate, clipped to the outer shard shape at the boundary.
func innerShapeAt(coord, chunkShape, outerShape []int64) []int64 {
	out := make([]int64, len(coord))
	for i := range coord {
		start := coord[i] * chunkShape[i]
		end := start + chunkShape[i]
		if end > outerShape[i] {
			end = outerShape[i]
		}
		out[i] = end - start
	}
	return out
}

// nextCoord increments a row-major grid coordinate in place; returns false on
// overflow (i.e., we've enumerated all coordinates).
func nextCoord(coord, grid []int64) bool {
	for i := len(coord) - 1; i >= 0; i-- {
		coord[i]++
		if coord[i] < grid[i] {
			return true
		}
		coord[i] = 0
	}
	return false
}

// copyRegion copies an inner chunk (shape innerShape) into the outer shard
// buffer (shape outerShape) at the given grid coordinate. sc/dstRegion/
// srcRegion are caller-owned scratch sized for outerShape's rank, reused
// across the DecodeAB loop.
func copyRegion(outer []byte, outerShape []int64, coord, chunkShape, innerShape []int64, inner []byte, esz int, sc *ndcopy.Scratch, dstRegion, srcRegion []ndcopy.Region) {
	for i := range outerShape {
		origin := coord[i] * chunkShape[i]
		dstRegion[i] = ndcopy.Region{Start: origin, Stop: origin + innerShape[i], Step: 1}
		srcRegion[i] = ndcopy.Region{Start: 0, Stop: innerShape[i], Step: 1}
	}
	ndcopy.Copy(outer, outerShape, dstRegion, inner, innerShape, srcRegion, esz, sc)
}

// extractRegion is the inverse of copyRegion: it returns a fresh copy of an
// inner chunk's bytes from the outer shard buffer. Scratch as for copyRegion.
func extractRegion(outer []byte, outerShape []int64, coord, chunkShape, innerShape []int64, esz int, sc *ndcopy.Scratch, dstRegion, srcRegion []ndcopy.Region) []byte {
	out := make([]byte, elementCount(innerShape)*int64(esz))
	for i := range outerShape {
		origin := coord[i] * chunkShape[i]
		srcRegion[i] = ndcopy.Region{Start: origin, Stop: origin + innerShape[i], Step: 1}
		dstRegion[i] = ndcopy.Region{Start: 0, Stop: innerShape[i], Step: 1}
	}
	ndcopy.Copy(out, innerShape, dstRegion, outer, outerShape, srcRegion, esz, sc)
	return out
}

// isZero reports whether b is all zero bytes.
func isZero(b []byte) bool {
	for _, x := range b {
		if x != 0 {
			return false
		}
	}
	return true
}

// writeFill replicates fill (one element's bytes) across dst in place. With
// fill nil/empty dst is zeroed. Doubling-copy keeps the inner loop log(n)
// rather than n/esz, and avoids an n-byte scratch allocation.
func writeFill(dst, fill []byte) {
	if len(fill) == 0 || isZero(fill) {
		clear(dst)
		return
	}
	if len(dst) < len(fill) {
		return
	}
	copy(dst[:len(fill)], fill)
	for filled := len(fill); filled < len(dst); filled *= 2 {
		copy(dst[filled:], dst[:filled])
	}
}

// matchesFill reports whether b is equal to fill replicated to len(b) bytes.
// Empty/nil fill is treated as all-zeros.
func matchesFill(b, fill []byte) bool {
	if len(fill) == 0 {
		return isZero(b)
	}
	if len(b)%len(fill) != 0 {
		return false
	}
	for i := 0; i < len(b); i += len(fill) {
		for j := 0; j < len(fill); j++ {
			if b[i+j] != fill[j] {
				return false
			}
		}
	}
	return true
}

// DecodeAB decodes a shard buffer into out.Data (which is sized to the outer
// shard shape). Inner chunks marked absent in the index are pre-filled with
// out.Fill (or zero when out.Fill is nil), matching the array's fill_value.
func (c *Codec) DecodeAB(in []byte, out *codec.ArrayChunk) error {
	if out == nil {
		return fmt.Errorf("sharding_indexed: nil output chunk")
	}
	esz := codec.DtypeSize(out.Dtype)
	if esz == 0 {
		return fmt.Errorf("sharding_indexed: unsupported dtype %q", out.Dtype)
	}

	grid, err := gridDims(out.Shape, c.chunkShape)
	if err != nil {
		return err
	}
	n := numInner(grid)

	idxLen, err := c.indexEncodedSize(n)
	if err != nil {
		return fmt.Errorf("sharding_indexed: compute index size: %w", err)
	}
	if int64(len(in)) < idxLen {
		return fmt.Errorf("sharding_indexed: shard too short: %d < index trailer %d", len(in), idxLen)
	}
	var idxBytes []byte
	switch c.indexLocation {
	case indexEnd:
		idxBytes = in[int64(len(in))-idxLen:]
	case indexStart:
		idxBytes = in[:idxLen]
	default:
		return fmt.Errorf("sharding_indexed: unknown index_location %d", c.indexLocation)
	}
	entries, err := c.decodeIndex(idxBytes, n)
	if err != nil {
		return err
	}

	totalElems := elementCount(out.Shape)
	totalBytes := totalElems * int64(esz)
	if int64(cap(out.Data)) >= totalBytes {
		out.Data = out.Data[:totalBytes]
	} else {
		out.Data = make([]byte, totalBytes)
	}
	// Pre-fill so absent inner chunks read back as the fill value.
	fill := out.Fill
	if len(fill) != esz {
		fill = nil
	}
	writeFill(out.Data, fill)

	rank := len(grid)
	ndSc := ndcopy.NewScratch(rank)
	dstRegion := make([]ndcopy.Region, rank)
	srcRegion := make([]ndcopy.Region, rank)
	coord := make([]int64, rank)
	for i := int64(0); i < n; i++ {
		off := entries[2*i]
		ln := entries[2*i+1]
		if off == emptyMarker && ln == emptyMarker {
			// Absent inner chunk: out.Data is already pre-filled with fill.
			if !nextCoord(coord, grid) && i != n-1 {
				return fmt.Errorf("sharding_indexed: ran out of grid coords")
			}
			continue
		}
		if off > uint64(len(in)) || ln > uint64(len(in)) || off+ln > uint64(len(in)) {
			return fmt.Errorf("sharding_indexed: index entry %d out of bounds (off=%d len=%d shard=%d)",
				i, off, ln, len(in))
		}
		innerShape := innerShapeAt(coord, c.chunkShape, out.Shape)
		innerChunk := codec.ArrayChunk{
			Shape: innerShape,
			Dtype: out.Dtype,
			Fill:  out.Fill,
		}
		if err := c.inner.Decode(in[off:off+ln], &innerChunk); err != nil {
			return fmt.Errorf("sharding_indexed: decode inner chunk %d: %w", i, err)
		}
		expected := elementCount(innerShape) * int64(esz)
		if int64(len(innerChunk.Data)) != expected {
			return fmt.Errorf("sharding_indexed: inner chunk %d data size %d != expected %d",
				i, len(innerChunk.Data), expected)
		}
		copyRegion(out.Data, out.Shape, coord, c.chunkShape, innerShape, innerChunk.Data, esz, ndSc, dstRegion, srcRegion)
		if !nextCoord(coord, grid) && i != n-1 {
			return fmt.Errorf("sharding_indexed: ran out of grid coords")
		}
	}
	return nil
}

// EncodeAB encodes an outer shard chunk into the on-wire shard byte buffer.
//
// Inner chunks whose element bytes equal in.Fill (replicated across the inner
// chunk) are written as empty (sentinel) entries in the index, matching the
// Zarr convention that fill-value regions are not stored. When in.Fill is nil
// the codec falls back to "all-zero means absent".
//
// The configured index_location controls whether the index trailer sits
// before ("start") or after ("end", default) the body of inner-chunk bytes.
func (c *Codec) EncodeAB(in codec.ArrayChunk) ([]byte, error) {
	esz := codec.DtypeSize(in.Dtype)
	if esz == 0 {
		return nil, fmt.Errorf("sharding_indexed: unsupported dtype %q", in.Dtype)
	}

	grid, err := gridDims(in.Shape, c.chunkShape)
	if err != nil {
		return nil, err
	}
	n := numInner(grid)
	expected := elementCount(in.Shape) * int64(esz)
	if int64(len(in.Data)) != expected {
		return nil, fmt.Errorf("sharding_indexed: input data size %d != expected %d",
			len(in.Data), expected)
	}

	// The index size (deterministic for fixed n) is needed up front to lay
	// out body offsets in the indexStart layout.
	idxLen, err := c.indexEncodedSize(n)
	if err != nil {
		return nil, fmt.Errorf("sharding_indexed: compute index size: %w", err)
	}
	var bodyOffset int64
	if c.indexLocation == indexStart {
		bodyOffset = idxLen
	}

	entries := make([]uint64, n*2)
	// Encoded inner chunks rarely exceed their raw size, so pre-sizing to
	// the raw size avoids append reallocations for large shards.
	body := make([]byte, 0, len(in.Data))

	fill := in.Fill
	if fill != nil && len(fill) != esz {
		return nil, fmt.Errorf("sharding_indexed: fill size %d != element size %d", len(fill), esz)
	}

	rank := len(grid)
	ndSc := ndcopy.NewScratch(rank)
	dstRegion := make([]ndcopy.Region, rank)
	srcRegion := make([]ndcopy.Region, rank)
	coord := make([]int64, rank)
	for i := int64(0); i < n; i++ {
		innerShape := innerShapeAt(coord, c.chunkShape, in.Shape)
		innerBytes := extractRegion(in.Data, in.Shape, coord, c.chunkShape, innerShape, esz, ndSc, dstRegion, srcRegion)
		if matchesFill(innerBytes, fill) {
			entries[2*i] = emptyMarker
			entries[2*i+1] = emptyMarker
		} else {
			innerChunk := codec.ArrayChunk{
				Shape: innerShape,
				Dtype: in.Dtype,
				Data:  innerBytes,
				Fill:  fill,
			}
			encoded, err := c.inner.Encode(innerChunk)
			if err != nil {
				return nil, fmt.Errorf("sharding_indexed: encode inner chunk %d: %w", i, err)
			}
			entries[2*i] = uint64(bodyOffset + int64(len(body)))
			entries[2*i+1] = uint64(len(encoded))
			body = append(body, encoded...)
		}
		if !nextCoord(coord, grid) && i != n-1 {
			return nil, fmt.Errorf("sharding_indexed: ran out of grid coords")
		}
	}

	indexBytes, err := c.encodeIndexEntries(entries)
	if err != nil {
		return nil, err
	}
	if int64(len(indexBytes)) != idxLen {
		// Should be impossible — encodeIndex is deterministic for fixed n.
		return nil, fmt.Errorf("sharding_indexed: index payload size %d != expected %d",
			len(indexBytes), idxLen)
	}
	out := make([]byte, 0, len(body)+len(indexBytes))
	switch c.indexLocation {
	case indexStart:
		out = append(out, indexBytes...)
		out = append(out, body...)
	default: // indexEnd
		out = append(out, body...)
		out = append(out, indexBytes...)
	}
	return out, nil
}

func init() {
	codec.Register("sharding_indexed", func(raw json.RawMessage) (codec.Codec, error) {
		c, err := New(raw)
		if err != nil {
			return nil, err // not a typed-nil codec.Codec
		}
		return c, nil
	})
}
