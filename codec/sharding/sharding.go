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

	"github.com/rkm/zarr-go/codec"
)

// emptyMarker is the (offset, length) value used in the index for an inner
// chunk that is absent (logically all fill-value).
const emptyMarker uint64 = math.MaxUint64

// IndexLocation indicates whether the index trailer is stored at the start or
// end of the shard.
type IndexLocation uint8

const (
	// IndexEnd places the index at the end of the shard (default).
	IndexEnd IndexLocation = iota
	// IndexStart places the index at the start of the shard.
	IndexStart
)

// Codec is the v3 "sharding_indexed" array→bytes codec.
type Codec struct {
	ChunkShape    []int64
	IndexLocation IndexLocation

	inner *codec.Pipeline
	index *codec.Pipeline
}

type configJSON struct {
	ChunkShape    []int64       `json:"chunk_shape"`
	Codecs        []codec.Spec  `json:"codecs"`
	IndexCodecs   []codec.Spec  `json:"index_codecs,omitempty"`
	IndexLocation *string       `json:"index_location,omitempty"`
}

// Name returns the v3 codec name.
func (*Codec) Name() string { return "sharding_indexed" }

// Configure parses the codec configuration and builds the inner and index
// pipelines.
func (c *Codec) Configure(raw json.RawMessage) error {
	if len(raw) == 0 {
		return fmt.Errorf("sharding_indexed: missing configuration")
	}
	var cfg configJSON
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return fmt.Errorf("sharding_indexed: invalid configuration: %w", err)
	}
	if len(cfg.ChunkShape) == 0 {
		return fmt.Errorf("sharding_indexed: chunk_shape is required")
	}
	for i, d := range cfg.ChunkShape {
		if d <= 0 {
			return fmt.Errorf("sharding_indexed: chunk_shape[%d]=%d must be positive", i, d)
		}
	}
	c.ChunkShape = append([]int64(nil), cfg.ChunkShape...)

	c.IndexLocation = IndexEnd
	if cfg.IndexLocation != nil {
		switch *cfg.IndexLocation {
		case "end":
			c.IndexLocation = IndexEnd
		case "start":
			c.IndexLocation = IndexStart
		default:
			return fmt.Errorf("sharding_indexed: invalid index_location %q", *cfg.IndexLocation)
		}
	}

	if len(cfg.Codecs) == 0 {
		return fmt.Errorf("sharding_indexed: inner codecs is required")
	}
	inner, err := codec.BuildPipeline(cfg.Codecs)
	if err != nil {
		return fmt.Errorf("sharding_indexed: build inner pipeline: %w", err)
	}
	c.inner = inner

	indexSpecs := cfg.IndexCodecs
	if len(indexSpecs) == 0 {
		indexSpecs = defaultIndexCodecs()
	}
	index, err := codec.BuildPipeline(indexSpecs)
	if err != nil {
		return fmt.Errorf("sharding_indexed: build index pipeline: %w", err)
	}
	c.index = index

	return nil
}

// defaultIndexCodecs is the default index_codecs pipeline:
// bytes (little) + crc32c.
func defaultIndexCodecs() []codec.Spec {
	return []codec.Spec{
		{Name: "bytes", Configuration: json.RawMessage(`{"endian":"little"}`)},
		{Name: "crc32c"},
	}
}

// elemSize returns the byte size of one element of the named dtype.
func elemSize(dtype string) int {
	switch dtype {
	case "bool", "int8", "uint8":
		return 1
	case "int16", "uint16", "float16":
		return 2
	case "int32", "uint32", "float32":
		return 4
	case "int64", "uint64", "float64", "complex64":
		return 8
	case "complex128":
		return 16
	}
	return 0
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
// the trailer length added by the index pipeline.
func (c *Codec) indexEncodedSize(n int64) (int64, error) {
	dummy := make([]byte, n*16)
	chunk := codec.ArrayChunk{
		Shape: []int64{n * 2},
		Dtype: "uint64",
		Data:  dummy,
	}
	encoded, err := c.encodeIndex(chunk)
	if err != nil {
		return 0, err
	}
	return int64(len(encoded)), nil
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
		entries[i] = binary.LittleEndian.Uint64(out.Data[i*8:])
	}
	return entries, nil
}

// encodeIndexEntries encodes a flat (offset, length) table into shard bytes.
func (c *Codec) encodeIndexEntries(entries []uint64) ([]byte, error) {
	buf := make([]byte, len(entries)*8)
	for i, v := range entries {
		binary.LittleEndian.PutUint64(buf[i*8:], v)
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

// copyRegion copies an inner chunk's element bytes into the outer shard buffer
// at the given grid coordinate. Both buffers are row-major; the inner chunk has
// shape innerShape, the outer has shape outerShape, and elements are esz bytes.
func copyRegion(outer []byte, outerShape []int64, coord, chunkShape, innerShape []int64, inner []byte, esz int) {
	rank := len(outerShape)
	// Compute element strides.
	outerStrides := make([]int64, rank)
	innerStrides := make([]int64, rank)
	outerStrides[rank-1] = 1
	innerStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		outerStrides[i] = outerStrides[i+1] * outerShape[i+1]
		innerStrides[i] = innerStrides[i+1] * innerShape[i+1]
	}
	// Origin element offset of this inner chunk inside the outer buffer.
	var originElems int64
	for i := 0; i < rank; i++ {
		originElems += coord[i] * chunkShape[i] * outerStrides[i]
	}
	// Walk the inner chunk in row-major order, copying contiguous rows
	// (the innermost dim) at a time.
	rowElems := innerShape[rank-1]
	rowBytes := rowElems * int64(esz)
	pos := make([]int64, rank-1) // per-axis position within the inner chunk
	for {
		// Compute the element offsets for this row.
		var innerElemOff, outerElemOff int64
		for i := 0; i < rank-1; i++ {
			innerElemOff += pos[i] * innerStrides[i]
			outerElemOff += pos[i] * outerStrides[i]
		}
		outerElemOff += originElems
		copy(outer[outerElemOff*int64(esz):outerElemOff*int64(esz)+rowBytes],
			inner[innerElemOff*int64(esz):innerElemOff*int64(esz)+rowBytes])
		// Advance the position vector (row-major over the leading dims).
		if rank == 1 {
			return
		}
		i := rank - 2
		for i >= 0 {
			pos[i]++
			if pos[i] < innerShape[i] {
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

// extractRegion does the inverse of copyRegion: it pulls an inner chunk's
// bytes out of the outer shard buffer.
func extractRegion(outer []byte, outerShape []int64, coord, chunkShape, innerShape []int64, esz int) []byte {
	rank := len(outerShape)
	innerElems := elementCount(innerShape)
	out := make([]byte, innerElems*int64(esz))

	outerStrides := make([]int64, rank)
	innerStrides := make([]int64, rank)
	outerStrides[rank-1] = 1
	innerStrides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		outerStrides[i] = outerStrides[i+1] * outerShape[i+1]
		innerStrides[i] = innerStrides[i+1] * innerShape[i+1]
	}
	var originElems int64
	for i := 0; i < rank; i++ {
		originElems += coord[i] * chunkShape[i] * outerStrides[i]
	}
	rowElems := innerShape[rank-1]
	rowBytes := rowElems * int64(esz)
	pos := make([]int64, rank-1)
	for {
		var innerElemOff, outerElemOff int64
		for i := 0; i < rank-1; i++ {
			innerElemOff += pos[i] * innerStrides[i]
			outerElemOff += pos[i] * outerStrides[i]
		}
		outerElemOff += originElems
		copy(out[innerElemOff*int64(esz):innerElemOff*int64(esz)+rowBytes],
			outer[outerElemOff*int64(esz):outerElemOff*int64(esz)+rowBytes])
		if rank == 1 {
			return out
		}
		i := rank - 2
		for i >= 0 {
			pos[i]++
			if pos[i] < innerShape[i] {
				break
			}
			pos[i] = 0
			i--
		}
		if i < 0 {
			return out
		}
	}
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

// DecodeAB decodes a shard buffer into out.Data (which is sized to the outer
// shard shape). Missing inner chunks leave their region zero-filled.
func (c *Codec) DecodeAB(in []byte, out *codec.ArrayChunk) error {
	if out == nil {
		return fmt.Errorf("sharding_indexed: nil output chunk")
	}
	if c.inner == nil || c.index == nil {
		return fmt.Errorf("sharding_indexed: codec not configured")
	}
	if c.IndexLocation != IndexEnd {
		return fmt.Errorf("sharding_indexed: index_location=start not supported")
	}
	esz := elemSize(out.Dtype)
	if esz == 0 {
		return fmt.Errorf("sharding_indexed: unsupported dtype %q", out.Dtype)
	}

	grid, err := gridDims(out.Shape, c.ChunkShape)
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
	idxStart := int64(len(in)) - idxLen
	entries, err := c.decodeIndex(in[idxStart:], n)
	if err != nil {
		return err
	}

	totalElems := elementCount(out.Shape)
	totalBytes := totalElems * int64(esz)
	if int64(cap(out.Data)) >= totalBytes {
		out.Data = out.Data[:totalBytes]
		// Zero it out: missing chunks read as fill (zero here).
		for i := range out.Data {
			out.Data[i] = 0
		}
	} else {
		out.Data = make([]byte, totalBytes)
	}

	coord := make([]int64, len(grid))
	for i := int64(0); i < n; i++ {
		off := entries[2*i]
		ln := entries[2*i+1]
		if off == emptyMarker && ln == emptyMarker {
			// Empty inner chunk: leave zeros.
			if !nextCoord(coord, grid) && i != n-1 {
				return fmt.Errorf("sharding_indexed: ran out of grid coords")
			}
			continue
		}
		if off > uint64(len(in)) || ln > uint64(len(in)) || off+ln > uint64(len(in)) {
			return fmt.Errorf("sharding_indexed: index entry %d out of bounds (off=%d len=%d shard=%d)",
				i, off, ln, len(in))
		}
		innerShape := innerShapeAt(coord, c.ChunkShape, out.Shape)
		innerChunk := codec.ArrayChunk{
			Shape: innerShape,
			Dtype: out.Dtype,
		}
		if err := c.inner.Decode(in[off:off+ln], &innerChunk); err != nil {
			return fmt.Errorf("sharding_indexed: decode inner chunk %d: %w", i, err)
		}
		expected := elementCount(innerShape) * int64(esz)
		if int64(len(innerChunk.Data)) != expected {
			return fmt.Errorf("sharding_indexed: inner chunk %d data size %d != expected %d",
				i, len(innerChunk.Data), expected)
		}
		copyRegion(out.Data, out.Shape, coord, c.ChunkShape, innerShape, innerChunk.Data, esz)
		if !nextCoord(coord, grid) && i != n-1 {
			return fmt.Errorf("sharding_indexed: ran out of grid coords")
		}
	}
	return nil
}

// EncodeAB encodes an outer shard chunk into the on-wire shard byte buffer.
//
// Inner chunks whose element bytes are all zero are written as empty (sentinel)
// entries in the index, matching the common Zarr convention that fill-value
// regions are not stored. Callers that need different fill semantics should
// pre-fill the buffer with their fill value before encoding.
func (c *Codec) EncodeAB(in codec.ArrayChunk) ([]byte, error) {
	if c.inner == nil || c.index == nil {
		return nil, fmt.Errorf("sharding_indexed: codec not configured")
	}
	if c.IndexLocation != IndexEnd {
		return nil, fmt.Errorf("sharding_indexed: index_location=start not supported")
	}
	esz := elemSize(in.Dtype)
	if esz == 0 {
		return nil, fmt.Errorf("sharding_indexed: unsupported dtype %q", in.Dtype)
	}

	grid, err := gridDims(in.Shape, c.ChunkShape)
	if err != nil {
		return nil, err
	}
	n := numInner(grid)
	expected := elementCount(in.Shape) * int64(esz)
	if int64(len(in.Data)) != expected {
		return nil, fmt.Errorf("sharding_indexed: input data size %d != expected %d",
			len(in.Data), expected)
	}

	entries := make([]uint64, n*2)
	var body []byte

	coord := make([]int64, len(grid))
	for i := int64(0); i < n; i++ {
		innerShape := innerShapeAt(coord, c.ChunkShape, in.Shape)
		innerBytes := extractRegion(in.Data, in.Shape, coord, c.ChunkShape, innerShape, esz)
		if isZero(innerBytes) {
			entries[2*i] = emptyMarker
			entries[2*i+1] = emptyMarker
		} else {
			innerChunk := codec.ArrayChunk{
				Shape: innerShape,
				Dtype: in.Dtype,
				Data:  innerBytes,
			}
			encoded, err := c.encodeInner(innerChunk)
			if err != nil {
				return nil, fmt.Errorf("sharding_indexed: encode inner chunk %d: %w", i, err)
			}
			entries[2*i] = uint64(len(body))
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
	out := make([]byte, 0, len(body)+len(indexBytes))
	out = append(out, body...)
	out = append(out, indexBytes...)
	return out, nil
}

// encodeInner runs the inner pipeline in forward order on a single inner chunk.
// (The Pipeline type only exposes Decode publicly, so we implement Encode here.)
func (c *Codec) encodeInner(in codec.ArrayChunk) ([]byte, error) {
	cur := in
	for _, aa := range c.inner.AA {
		next := codec.ArrayChunk{Shape: cur.Shape, Dtype: cur.Dtype}
		if err := aa.EncodeAA(cur, &next); err != nil {
			return nil, fmt.Errorf("inner %s: %w", aa.Name(), err)
		}
		cur = next
	}
	buf, err := c.inner.AB.EncodeAB(cur)
	if err != nil {
		return nil, fmt.Errorf("inner %s: %w", c.inner.AB.Name(), err)
	}
	for _, bb := range c.inner.BB {
		buf, err = bb.EncodeBB(buf)
		if err != nil {
			return nil, fmt.Errorf("inner %s: %w", bb.Name(), err)
		}
	}
	return buf, nil
}

func init() {
	codec.Register("sharding_indexed", func() codec.Codec { return &Codec{} })
}
