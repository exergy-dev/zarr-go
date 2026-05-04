package zarr

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"sync"
	"unsafe"

	"github.com/rkm/zarr-go/codec"
	"github.com/rkm/zarr-go/internal/bufpool"
	"github.com/rkm/zarr-go/internal/metadata"
	"github.com/rkm/zarr-go/store"
)

// Array is the typed view of a Zarr array. Construct via OpenArray[T].
type Array[T Numeric] struct {
	store    store.Store
	path     string
	meta     *metadata.ArrayMeta
	pipeline *codec.Pipeline
	dtype    Dtype
	fill     []byte // fill value bytes, native LE, sizeof(T)
	opts     arrayOptions
}

// Chunk is one decoded chunk yielded from Stream.
type Chunk[T Numeric] struct {
	Coord []int64
	Shape []int64
	Data  []T
}

// Shape returns the array's logical shape.
func (a *Array[T]) Shape() []int64 {
	out := make([]int64, len(a.meta.Shape))
	copy(out, a.meta.Shape)
	return out
}

// Chunks returns the array's chunk shape.
func (a *Array[T]) Chunks() []int64 {
	cs := a.meta.ChunkGrid.Configuration.ChunkShape
	out := make([]int64, len(cs))
	copy(out, cs)
	return out
}

// Dtype returns the array's element type.
func (a *Array[T]) Dtype() Dtype { return a.dtype }

// Read materializes the selection into dst. dst must hold sel.NumElements()
// elements; layout is row-major (C order).
func (a *Array[T]) Read(ctx context.Context, sel Selection, dst []T) error {
	if err := sel.Validate(a.meta.Shape); err != nil {
		return err
	}
	want := sel.NumElements()
	if int64(len(dst)) != want {
		return fmt.Errorf("%w: dst has %d elements, selection wants %d",
			ErrShapeMismatch, len(dst), want)
	}
	// Pre-fill dst with the fill value so missing chunks are correct without
	// any extra work later. For zero fill (the common case) this is cheap.
	if hasNonZeroFill(a.fill) {
		fillTyped[T](dst, a.fill)
	} // else: Go zero-initializes

	chunkShape := a.meta.ChunkGrid.Configuration.ChunkShape
	rank := len(a.meta.Shape)
	outShape := make([]int64, rank)
	for i, r := range sel.Ranges {
		outShape[i] = r.Length()
	}
	outStrides := rowMajorStrides(outShape)

	work := make([]ChunkCoord, 0)
	for c := range ChunksIntersecting(a.meta.Shape, chunkShape, sel) {
		work = append(work, c.Clone())
	}
	if len(work) == 0 {
		return nil
	}

	conc := a.opts.concurrency
	if conc > len(work) {
		conc = len(work)
	}
	if conc < 1 {
		conc = 1
	}
	sem := make(chan struct{}, conc)
	var wg sync.WaitGroup
	var firstErrMu sync.Mutex
	var firstErr error
	cctx, cancel := context.WithCancel(ctx)
	defer cancel()
	setErr := func(err error) {
		firstErrMu.Lock()
		if firstErr == nil {
			firstErr = err
			cancel()
		}
		firstErrMu.Unlock()
	}

	for _, c := range work {
		coord := c
		select {
		case <-cctx.Done():
			break
		case sem <- struct{}{}:
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			key := metadata.ChunkKey(a.path, a.meta.ChunkKeyEnc, []int64(coord))
			raw, err := a.store.Get(cctx, key)
			if err != nil {
				if errors.Is(err, store.ErrKeyNotFound) {
					return // missing chunk = fill, already in dst
				}
				setErr(fmt.Errorf("fetch chunk %v: %w", []int64(coord), err))
				return
			}
			if err := a.decodeAndCopy(coord, raw, sel, chunkShape, outShape, outStrides, dst); err != nil {
				setErr(err)
			}
		}()
	}
	wg.Wait()
	if firstErr != nil {
		return firstErr
	}
	return nil
}

// decodeAndCopy decodes one already-fetched chunk and copies its sub-region
// into dst. Caller-supplied raw is the on-wire chunk bytes.
func (a *Array[T]) decodeAndCopy(
	coord ChunkCoord,
	raw []byte,
	sel Selection,
	chunkShape, outShape, outStrides []int64,
	dst []T,
) error {
	chunkSrc, outDst := ChunkSelection(a.meta.Shape, chunkShape, sel, coord)
	elemSize := int(unsafe.Sizeof(*new(T)))
	chunkBytes := elemSize
	for _, d := range chunkShape {
		chunkBytes *= int(d)
	}
	pooled := bufpool.Get(chunkBytes)
	defer bufpool.Put(pooled)
	chunk := codec.ArrayChunk{
		Shape: chunkShape, // safe to share; codec doesn't mutate
		Dtype: a.meta.DataType,
		Data:  pooled[:0],
	}
	if err := a.pipeline.Decode(raw, &chunk); err != nil {
		return &DecodeError{Coord: []int64(coord), Codec: a.pipeline.AB.Name(), Err: err}
	}

	copyChunkRegion[T](
		chunk.Data, chunkShape, chunkSrc,
		dst, outShape, outStrides, outDst,
		elemSize,
	)
	return nil
}

// Write encodes the elements in src and writes them to the chunks intersecting
// sel. src must hold sel.NumElements() elements; layout is row-major.
//
// Each touched chunk is read-modify-written: if the existing chunk exists it is
// fetched and merged with src in the intersecting region, otherwise a new
// chunk filled with the array's fill value is created. Chunks that the
// selection covers exactly (the whole chunk shape) skip the read.
func (a *Array[T]) Write(ctx context.Context, sel Selection, src []T) error {
	if err := sel.Validate(a.meta.Shape); err != nil {
		return err
	}
	want := sel.NumElements()
	if int64(len(src)) != want {
		return fmt.Errorf("%w: src has %d elements, selection wants %d",
			ErrShapeMismatch, len(src), want)
	}
	ws, ok := a.store.(store.WritableStore)
	if !ok {
		return errReadOnly
	}

	chunkShape := a.meta.ChunkGrid.Configuration.ChunkShape
	rank := len(a.meta.Shape)
	srcShape := make([]int64, rank)
	for i, r := range sel.Ranges {
		srcShape[i] = r.Length()
	}
	srcStrides := rowMajorStrides(srcShape)
	elemSize := int(unsafe.Sizeof(*new(T)))

	srcBytes := unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), len(src)*elemSize)

	for c := range ChunksIntersecting(a.meta.Shape, chunkShape, sel) {
		coord := c.Clone()
		if err := a.writeOneChunk(ctx, ws, coord, sel, chunkShape, srcShape, srcStrides, srcBytes, elemSize); err != nil {
			return err
		}
	}
	return nil
}

func (a *Array[T]) writeOneChunk(
	ctx context.Context,
	ws store.WritableStore,
	coord ChunkCoord,
	sel Selection,
	chunkShape, srcShape, srcStrides []int64,
	srcBytes []byte,
	elemSize int,
) error {
	chunkSrc, outDst := ChunkSelection(a.meta.Shape, chunkShape, sel, coord)
	rank := len(chunkShape)

	// Compute covered region per dim. If it equals the full chunk_shape
	// (clipped to the array), we can skip the read-modify path.
	totalChunkBytes := int64(elemSize)
	for _, d := range chunkShape {
		totalChunkBytes *= d
	}
	chunkBuf := make([]byte, totalChunkBytes)

	regionLen := make([]int64, rank)
	fullCovered := true
	for i := 0; i < rank; i++ {
		regionLen[i] = chunkSrc[i].Stop - chunkSrc[i].Start
		// "Full coverage" within array bounds: chunkSrc spans [0, in-array-extent)
		// AND step is 1.
		origin := coord[i] * chunkShape[i]
		end := origin + chunkShape[i]
		if end > a.meta.Shape[i] {
			end = a.meta.Shape[i]
		}
		inArrayExtent := end - origin
		if !(chunkSrc[i].Start == 0 && chunkSrc[i].Stop == inArrayExtent && chunkSrc[i].Step == 1) {
			fullCovered = false
		}
	}

	key := metadata.ChunkKey(a.path, a.meta.ChunkKeyEnc, []int64(coord))

	if !fullCovered {
		// Read-modify-write: try to load existing chunk.
		existing, err := a.store.Get(ctx, key)
		switch {
		case err == nil:
			// Decode existing into chunkBuf.
			ck := codec.ArrayChunk{Shape: append([]int64(nil), chunkShape...), Dtype: a.meta.DataType, Data: chunkBuf[:0]}
			if err := a.pipeline.Decode(existing, &ck); err != nil {
				return &DecodeError{Coord: []int64(coord), Codec: a.pipeline.AB.Name(), Err: err}
			}
			if int64(len(ck.Data)) != totalChunkBytes {
				return fmt.Errorf("decoded chunk size %d != expected %d", len(ck.Data), totalChunkBytes)
			}
			copy(chunkBuf, ck.Data)
		case errors.Is(err, store.ErrKeyNotFound):
			// Initialize chunkBuf with fill value bytes.
			fillChunk(chunkBuf, a.fill, elemSize)
		default:
			return fmt.Errorf("fetch chunk %v: %w", []int64(coord), err)
		}
	} else if hasNonZeroFill(a.fill) {
		// We're going to overwrite every in-array byte, but padded edge
		// positions still need fill.
		fillChunk(chunkBuf, a.fill, elemSize)
	}

	// Copy from src into chunkBuf at chunkSrc positions, drawing from outDst
	// positions in the user's src.
	chunkStrides := rowMajorStrides(chunkShape)
	total := int64(1)
	for _, n := range regionLen {
		total *= n
	}
	idx := make([]int64, rank)
	for k := int64(0); k < total; k++ {
		var srcEl, dstEl int64
		for d := 0; d < rank; d++ {
			srcCoord := outDst[d].Start + idx[d]
			srcEl += srcCoord * srcStrides[d]
			dstCoord := chunkSrc[d].Start + idx[d]*chunkSrc[d].Step
			dstEl += dstCoord * chunkStrides[d]
		}
		copy(
			chunkBuf[dstEl*int64(elemSize):(dstEl+1)*int64(elemSize)],
			srcBytes[srcEl*int64(elemSize):(srcEl+1)*int64(elemSize)],
		)
		for d := rank - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < regionLen[d] {
				break
			}
			idx[d] = 0
		}
	}

	encoded, err := a.pipeline.Encode(codec.ArrayChunk{
		Shape: append([]int64(nil), chunkShape...),
		Dtype: a.meta.DataType,
		Data:  chunkBuf,
	})
	if err != nil {
		return fmt.Errorf("encode chunk %v: %w", []int64(coord), err)
	}
	if err := ws.Set(ctx, key, encoded); err != nil {
		return fmt.Errorf("write chunk %v: %w", []int64(coord), err)
	}
	return nil
}

// fillChunk replicates one element's fill bytes across buf.
func fillChunk(buf, fill []byte, elemSize int) {
	if len(fill) != elemSize || elemSize == 0 {
		return
	}
	if !hasNonZeroFill(fill) {
		// already zeros from make
		return
	}
	for i := 0; i < len(buf); i += elemSize {
		copy(buf[i:i+elemSize], fill)
	}
}

// Stream yields decoded chunks intersecting sel without assembling the full
// output buffer.
func (a *Array[T]) Stream(ctx context.Context, sel Selection) iter.Seq2[Chunk[T], error] {
	return func(yield func(Chunk[T], error) bool) {
		if err := sel.Validate(a.meta.Shape); err != nil {
			yield(Chunk[T]{}, err)
			return
		}
		chunkShape := a.meta.ChunkGrid.Configuration.ChunkShape
		elemSize := int(unsafe.Sizeof(*new(T)))
		for c := range ChunksIntersecting(a.meta.Shape, chunkShape, sel) {
			coord := c.Clone()
			actualChunk := make([]int64, len(chunkShape))
			for i, cs := range chunkShape {
				origin := coord[i] * cs
				end := origin + cs
				if end > a.meta.Shape[i] {
					end = a.meta.Shape[i]
				}
				actualChunk[i] = end - origin
			}
			key := metadata.ChunkKey(a.path, a.meta.ChunkKeyEnc, []int64(coord))
			raw, err := a.store.Get(ctx, key)
			if err != nil {
				if errors.Is(err, store.ErrKeyNotFound) {
					// Synthetic fill-value chunk.
					n := int64(1)
					for _, d := range actualChunk {
						n *= d
					}
					data := make([]T, n)
					if hasNonZeroFill(a.fill) {
						fillTyped[T](data, a.fill)
					}
					if !yield(Chunk[T]{Coord: []int64(coord), Shape: actualChunk, Data: data}, nil) {
						return
					}
					continue
				}
				yield(Chunk[T]{Coord: []int64(coord)}, err)
				return
			}
			ck := codec.ArrayChunk{Shape: append([]int64(nil), chunkShape...), Dtype: a.meta.DataType}
			if err := a.pipeline.Decode(raw, &ck); err != nil {
				yield(Chunk[T]{Coord: []int64(coord)}, &DecodeError{
					Coord: []int64(coord), Codec: a.pipeline.AB.Name(), Err: err,
				})
				return
			}
			n := len(ck.Data) / elemSize
			data := unsafe.Slice((*T)(unsafe.Pointer(&ck.Data[0])), n)
			if !yield(Chunk[T]{Coord: []int64(coord), Shape: actualChunk, Data: data}, nil) {
				return
			}
		}
	}
}

// rowMajorStrides returns element strides (not byte strides) in row-major order.
func rowMajorStrides(shape []int64) []int64 {
	rank := len(shape)
	s := make([]int64, rank)
	if rank == 0 {
		return s
	}
	s[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		s[i] = s[i+1] * shape[i+1]
	}
	return s
}

// hasNonZeroFill reports whether any byte in fill is non-zero.
func hasNonZeroFill(fill []byte) bool {
	for _, b := range fill {
		if b != 0 {
			return true
		}
	}
	return false
}

// fillTyped writes the fill value (one element's worth of native-LE bytes)
// across every element of dst.
func fillTyped[T Numeric](dst []T, fill []byte) {
	if len(dst) == 0 || len(fill) == 0 {
		return
	}
	var v T
	dstBytes := unsafe.Slice((*byte)(unsafe.Pointer(&v)), int(unsafe.Sizeof(v)))
	copy(dstBytes, fill)
	for i := range dst {
		dst[i] = v
	}
}

// copyChunkRegion copies the region [chunkSrc] of the chunk's element bytes
// into the region [outDst] of dst (a row-major typed slice with strides
// outStrides). It iterates the row-major positions in the destination region;
// for each, computes the source element offset by walking stride-aware.
func copyChunkRegion[T Numeric](
	chunkBytes []byte, chunkShape []int64, chunkSrc []DimRange,
	dst []T, outShape, outStrides []int64, outDst []DimRange,
	elemSize int,
) {
	rank := len(chunkShape)
	chunkStrides := rowMajorStrides(chunkShape)
	regionLen := make([]int64, rank)
	for i := range regionLen {
		regionLen[i] = outDst[i].Stop - outDst[i].Start
	}
	total := int64(1)
	for _, n := range regionLen {
		total *= n
	}
	if total == 0 {
		return
	}
	dstBytes := unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*elemSize)

	idx := make([]int64, rank)
	for k := int64(0); k < total; k++ {
		// Map idx → source element offset, dest element offset.
		var srcEl, dstEl int64
		for d := 0; d < rank; d++ {
			srcCoord := chunkSrc[d].Start + idx[d]*chunkSrc[d].Step
			srcEl += srcCoord * chunkStrides[d]
			dstCoord := outDst[d].Start + idx[d]
			dstEl += dstCoord * outStrides[d]
		}
		copy(
			dstBytes[dstEl*int64(elemSize):(dstEl+1)*int64(elemSize)],
			chunkBytes[srcEl*int64(elemSize):(srcEl+1)*int64(elemSize)],
		)
		// Increment idx in row-major order (last dim fastest).
		for d := rank - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < regionLen[d] {
				break
			}
			idx[d] = 0
		}
	}
}
