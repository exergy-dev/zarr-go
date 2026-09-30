package zarr

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"sync"
	"unsafe"

	"github.com/exergy-dev/zarr-go/codec"
	bytescodec "github.com/exergy-dev/zarr-go/codec/bytes"
	"github.com/exergy-dev/zarr-go/internal/bufpool"
	"github.com/exergy-dev/zarr-go/internal/metadata"
	"github.com/exergy-dev/zarr-go/internal/ndcopy"
	"github.com/exergy-dev/zarr-go/store"
)

// hostLittleEndian is true if the host CPU is little-endian.
var hostLittleEndian = func() bool {
	var x uint16 = 1
	return *(*byte)(unsafe.Pointer(&x)) == 1
}()

// pipelinePartialDecoder returns the AB codec as a codec.PartialDecoder when
// it implements one and no BB codecs sit on top of it (a BB codec would have
// to decode the whole blob first), else nil.
func pipelinePartialDecoder(p *codec.Pipeline) codec.PartialDecoder {
	if p == nil || p.AB == nil || len(p.BB) != 0 {
		return nil
	}
	pd, ok := p.AB.(codec.PartialDecoder)
	if !ok {
		return nil
	}
	return pd
}

// isRawCopyFastPipeline reports whether the pipeline is exactly
// [bytes(<host endian>)] with no AA or BB codecs, so Pipeline.Decode would be
// a plain copy and the raw chunk bytes can feed ndcopy.Copy directly.
func isRawCopyFastPipeline(p *codec.Pipeline) bool {
	if p == nil || p.AB == nil {
		return false
	}
	if len(p.AA) != 0 || len(p.BB) != 0 {
		return false
	}
	bc, ok := p.AB.(*bytescodec.Codec)
	if !ok {
		return false
	}
	if hostLittleEndian {
		return bc.Endian() == bytescodec.Little
	}
	return bc.Endian() == bytescodec.Big
}

// Array is the typed view of a Zarr array. Construct via OpenArray[T].
//
// Concurrency: Array[T].Read is internally concurrent (it dispatches per-chunk
// fetch+decode goroutines bounded by WithConcurrency / WithIOConcurrency) and
// is safe to call from multiple goroutines on the same Array.
//
// Array[T].Write performs read-modify-write on each touched chunk and does
// NOT take a chunk-level lock. Concurrent Writes that touch the same chunk
// can lose updates; concurrent Read+Write may observe a torn intermediate
// state. Callers that need either pattern must serialize at the application
// layer (one goroutine writing per array, or a per-chunk mutex).
type Array[T Numeric] struct {
	store    store.Store
	mmap     store.MmapStore // non-nil iff store implements MmapStore
	path     string
	meta     *metadata.ArrayMeta
	pipeline *codec.Pipeline
	dtype    Dtype
	fill     []byte // fill value bytes, native byte order, sizeof(T)
	opts     arrayOptions
	// partialDecoder and rawCopyFast are derived from pipeline once at open
	// (see pipelinePartialDecoder and isRawCopyFastPipeline).
	partialDecoder codec.PartialDecoder
	rawCopyFast    bool
}

// Chunk is one decoded chunk yielded from Stream. Shape is clipped to the
// array bounds and Data holds exactly product(Shape) elements in row-major
// order. Data is freshly allocated for each chunk and owned by the caller.
type Chunk[T Numeric] struct {
	Coord []int64
	Shape []int64
	Data  []T
}

// Shape returns the array's logical shape.
func (a *Array[T]) Shape() []int64 { return metadata.CloneInt64(a.meta.Shape) }

// Chunks returns the array's chunk shape.
func (a *Array[T]) Chunks() []int64 {
	return metadata.CloneInt64(a.meta.ChunkGrid.Configuration.ChunkShape)
}

// Dtype returns the array's element type.
func (a *Array[T]) Dtype() Dtype { return a.dtype }

// Attrs returns a copy of the array's user attributes (v3 "attributes", or
// the v2 .zattrs sidecar), or nil if it has none.
func (a *Array[T]) Attrs() map[string]any { return cloneAttrs(a.meta.Attributes) }

// DimensionNames returns a copy of the array's v3 dimension_names, or nil
// if it has none (always nil for v2 arrays).
func (a *Array[T]) DimensionNames() []string { return slices.Clone(a.meta.Dimensions) }

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
	if want == 0 {
		return nil // empty selection: no chunk intersects it
	}
	// Pre-fill dst so missing chunks need no further work.
	if hasNonZeroFill(a.fill) {
		fillTyped[T](dst, a.fill)
	} // else: Go zero-initializes

	chunkShape := a.meta.ChunkGrid.Configuration.ChunkShape
	outShape := make([]int64, len(a.meta.Shape))
	for i, r := range sel.Ranges {
		outShape[i] = r.Length()
	}

	// ioSem bounds in-flight store fetches (and, for the partial-decode path,
	// the whole fetch+inner-decode operation, which is I/O-bound overall).
	// decodeSem separately bounds the CPU-bound full-chunk decode step, so a
	// high ioConcurrency (good for a network-bound store) doesn't also
	// oversubscribe CPU decode workers.
	ioSem := make(chan struct{}, a.opts.ioConcurrency)
	decodeSem := make(chan struct{}, a.opts.concurrency)
	var wg sync.WaitGroup
	var firstErrMu sync.Mutex
	var firstErr error
	cctx, cancel := context.WithCancel(ctx)
	defer cancel()

launch:
	for c := range ChunksIntersecting(a.meta.Shape, chunkShape, sel) {
		select {
		case <-cctx.Done():
			break launch
		case ioSem <- struct{}{}:
		}
		coord := c.Clone()
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-ioSem }()
			if err := a.readChunk(cctx, coord, sel, chunkShape, outShape, dst, decodeSem); err != nil {
				firstErrMu.Lock()
				if firstErr == nil {
					firstErr = err
					cancel()
				}
				firstErrMu.Unlock()
			}
		}()
	}
	wg.Wait()
	if firstErr == nil {
		// If ctx was cancelled, chunks may have been skipped without any
		// goroutine recording an error; returning nil would silently hand
		// back incomplete data.
		firstErr = context.Cause(ctx)
	}
	return firstErr
}

// readChunk fetches, decodes and copies into dst the part of one chunk that
// sel covers. A missing chunk is not an error: dst already holds the fill
// value there. decodeSem bounds the CPU-bound full-chunk decode step.
func (a *Array[T]) readChunk(
	ctx context.Context,
	coord ChunkCoord,
	sel Selection,
	chunkShape, outShape []int64,
	dst []T,
	decodeSem chan struct{},
) error {
	key := metadata.ChunkKey(a.path, a.meta.ChunkKeyEnc, []int64(coord))
	if a.partialDecoder != nil {
		// Interleaves I/O and per-inner-chunk decode internally; runs
		// entirely under the caller's I/O gate, no decodeSem.
		err := a.partialDecodeAndCopy(ctx, a.partialDecoder, coord, key, sel, chunkShape, outShape, dst)
		if errors.Is(err, store.ErrKeyNotFound) {
			return nil
		}
		return err
	}
	var (
		raw     []byte
		release func()
		err     error
	)
	if a.mmap != nil {
		raw, release, err = a.mmap.GetMmap(ctx, key)
	} else {
		raw, err = a.store.Get(ctx, key)
	}
	if errors.Is(err, store.ErrKeyNotFound) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("fetch chunk %v: %w", []int64(coord), err)
	}
	if release != nil {
		defer release()
	}
	select {
	case <-ctx.Done():
		return nil // Read reports the error or cancellation cause
	case decodeSem <- struct{}{}:
	}
	defer func() { <-decodeSem }()
	return a.decodeAndCopy(coord, raw, sel, chunkShape, outShape, dst)
}

// chunkActualShape returns the (possibly clipped) extent of the chunk at
// coord, i.e. the chunk's shape after clipping against the array bounds at
// edge chunks.
func (a *Array[T]) chunkActualShape(coord ChunkCoord, chunkShape []int64, out []int64) {
	for i, cs := range chunkShape {
		origin := coord[i] * cs
		end := origin + cs
		if end > a.meta.Shape[i] {
			end = a.meta.Shape[i]
		}
		out[i] = end - origin
	}
}

// partialDecodeAndCopy issues a partial-decode request for the requested
// sub-region of the chunk at coord and copies the result into dst.
//
// If the selection covers the entire (clipped) chunk or any dim has
// step != 1, it falls back to a full fetch and Pipeline.Decode.
func (a *Array[T]) partialDecodeAndCopy(
	ctx context.Context,
	pd codec.PartialDecoder,
	coord ChunkCoord,
	key string,
	sel Selection,
	chunkShape, outShape []int64,
	dst []T,
) error {
	rank := len(chunkShape)
	chunkSrc, outDst := chunkSelection(a.meta.Shape, chunkShape, sel, coord)

	// Partial decode requires step==1; bail to the full path if any dim is stepped.
	actual := make([]int64, rank)
	a.chunkActualShape(coord, chunkShape, actual)
	isFull := true
	for d := 0; d < rank; d++ {
		if chunkSrc[d].Step != 1 {
			// Stepped: full fetch + decode handles striding correctly.
			raw, err := a.store.Get(ctx, key)
			if err != nil {
				return err
			}
			return a.decodeAndCopy(coord, raw, sel, chunkShape, outShape, dst)
		}
		if chunkSrc[d].Start != 0 || chunkSrc[d].Stop != actual[d] {
			isFull = false
		}
	}
	if isFull {
		// Whole-chunk read — partial would just degenerate, do the regular path.
		raw, err := a.store.Get(ctx, key)
		if err != nil {
			return err
		}
		return a.decodeAndCopy(coord, raw, sel, chunkShape, outShape, dst)
	}

	elemSize := int(unsafe.Sizeof(*new(T)))
	regionShape := make([]int64, rank)
	regionElems := int64(1)
	for d := 0; d < rank; d++ {
		regionShape[d] = chunkSrc[d].Stop - chunkSrc[d].Start
		regionElems *= regionShape[d]
	}
	regionBytes := regionElems * int64(elemSize)
	pooled := bufpool.Get(int(regionBytes))
	defer bufpool.Put(pooled)
	out := pooled[:regionBytes]

	// fullShape is the FULL configured chunk shape, never the array-bounds-
	// clipped actual: chunks (including shards) are always encoded at the
	// full chunk_shape, so feeding a clipped shape here would compute the
	// wrong inner chunk grid (wrong index size, garbage entries). actual is
	// only used above to size the step==1/isFull short-circuit checks.
	req := codec.PartialReadRequest{
		Key:       key,
		FullShape: chunkShape,
		Dtype:     a.meta.DataType,
		Region:    chunkSrc, // every Step is 1 (checked above)
		Out:       out,
		Fill:      a.fill,
		GapBytes:  a.opts.rangeGap,
	}
	if err := pd.DecodePartial(ctx, a.store, &req); err != nil {
		// Pass ErrKeyNotFound through unwrapped so the caller leaves the
		// fill value in place, as on the full path.
		if errors.Is(err, store.ErrKeyNotFound) {
			return err
		}
		return &DecodeError{Coord: []int64(coord), Err: fmt.Errorf("codec %q: %w", a.pipeline.AB.Name(), err)}
	}

	// The dense region buffer maps 1:1 onto outDst (Step==1 along every
	// dim), so copy it as a whole "chunk" of shape regionShape.
	whole := make([]ndcopy.Region, rank)
	for d := 0; d < rank; d++ {
		whole[d] = ndcopy.Region{Start: 0, Stop: regionShape[d], Step: 1}
	}
	var dstBytes []byte
	if len(dst) > 0 {
		dstBytes = unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*elemSize)
	}
	ndcopy.Copy(dstBytes, outShape, outDst, out, regionShape, whole, elemSize, nil)
	return nil
}

// decodeAndCopy decodes one already-fetched chunk and copies its sub-region
// into dst. Caller-supplied raw is the on-wire chunk bytes.
func (a *Array[T]) decodeAndCopy(
	coord ChunkCoord,
	raw []byte,
	sel Selection,
	chunkShape, outShape []int64,
	dst []T,
) error {
	chunkSrc, outDst := chunkSelection(a.meta.Shape, chunkShape, sel, coord)
	elemSize := int(unsafe.Sizeof(*new(T)))
	chunkBytes := elemSize
	for _, d := range chunkShape {
		chunkBytes *= int(d)
	}
	var dstBytes []byte
	if len(dst) > 0 {
		dstBytes = unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*elemSize)
	}

	// Raw bytes are already native element bytes: skip Decode and the pool.
	if a.rawCopyFast && len(raw) == chunkBytes {
		ndcopy.Copy(dstBytes, outShape, outDst, raw, chunkShape, chunkSrc, elemSize, nil)
		return nil
	}

	pooled := bufpool.Get(chunkBytes)
	defer bufpool.Put(pooled)
	chunk := codec.ArrayChunk{
		Shape: chunkShape, // safe to share; codec doesn't mutate
		Dtype: a.meta.DataType,
		Data:  pooled[:0],
		Fill:  a.fill,
	}
	if err := a.pipeline.Decode(raw, &chunk); err != nil {
		return &DecodeError{Coord: []int64(coord), Err: err}
	}
	if len(chunk.Data) != chunkBytes {
		return &DecodeError{Coord: []int64(coord), Err: fmt.Errorf("decoded chunk size %d != expected %d", len(chunk.Data), chunkBytes)}
	}

	ndcopy.Copy(dstBytes, outShape, outDst, chunk.Data, chunkShape, chunkSrc, elemSize, nil)
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
	ws, ok := a.store.(store.Writer)
	if !ok {
		return ErrReadOnly
	}
	if want == 0 {
		return nil // empty selection: nothing to write, and &src[0] below would panic
	}

	chunkShape := a.meta.ChunkGrid.Configuration.ChunkShape
	rank := len(a.meta.Shape)
	srcShape := make([]int64, rank)
	for i, r := range sel.Ranges {
		srcShape[i] = r.Length()
	}
	elemSize := int(unsafe.Sizeof(*new(T)))

	srcBytes := unsafe.Slice((*byte)(unsafe.Pointer(&src[0])), len(src)*elemSize)

	for c := range ChunksIntersecting(a.meta.Shape, chunkShape, sel) {
		coord := c.Clone()
		if err := a.writeOneChunk(ctx, ws, coord, sel, chunkShape, srcShape, srcBytes, elemSize); err != nil {
			return err
		}
	}
	return nil
}

func (a *Array[T]) writeOneChunk(
	ctx context.Context,
	ws store.Writer,
	coord ChunkCoord,
	sel Selection,
	chunkShape, srcShape []int64,
	srcBytes []byte,
	elemSize int,
) error {
	rank := len(chunkShape)
	// chunkSel is the chunk side (the write destination; may be stepped),
	// srcSel the user src side (the copy source; always step 1).
	chunkSel, srcSel := chunkSelection(a.meta.Shape, chunkShape, sel, coord)

	totalChunkBytes := int64(elemSize)
	for _, d := range chunkShape {
		totalChunkBytes *= d
	}
	chunkBuf := bufpool.Get(int(totalChunkBytes))
	defer bufpool.Put(chunkBuf)

	// fullCovered (in-array-extent based) gates whether the existing chunk
	// needs to be read at all: if the write covers every in-array-visible
	// element, there's nothing worth preserving from the old chunk.
	// noPadding (full chunkShape based) additionally requires there be no
	// clipped edge region — only then does the copy below touch every byte
	// of chunkBuf, making a fill pass beforehand pure waste.
	fullCovered := true
	noPadding := true
	for i := 0; i < rank; i++ {
		origin := coord[i] * chunkShape[i]
		end := origin + chunkShape[i]
		if end > a.meta.Shape[i] {
			end = a.meta.Shape[i]
		}
		inArrayExtent := end - origin
		if chunkSel[i].Start != 0 || chunkSel[i].Stop != inArrayExtent || chunkSel[i].Step != 1 {
			fullCovered = false
		}
		if inArrayExtent != chunkShape[i] {
			noPadding = false
		}
	}

	key := metadata.ChunkKey(a.path, a.meta.ChunkKeyEnc, []int64(coord))

	if !fullCovered {
		existing, err := a.store.Get(ctx, key)
		switch {
		case err == nil:
			ck := codec.ArrayChunk{Shape: append([]int64(nil), chunkShape...), Dtype: a.meta.DataType, Data: chunkBuf[:0], Fill: a.fill}
			if err := a.pipeline.Decode(existing, &ck); err != nil {
				return &DecodeError{Coord: []int64(coord), Err: err}
			}
			if int64(len(ck.Data)) != totalChunkBytes {
				return fmt.Errorf("decoded chunk size %d != expected %d", len(ck.Data), totalChunkBytes)
			}
			copy(chunkBuf, ck.Data)
		case errors.Is(err, store.ErrKeyNotFound):
			fillChunk(chunkBuf, a.fill, elemSize)
		default:
			return fmt.Errorf("fetch chunk %v: %w", []int64(coord), err)
		}
	} else if !noPadding || hasNonZeroFill(a.fill) {
		// Either padded edge positions need fill, or every in-array byte is
		// about to be overwritten but chunkBuf (pooled — possibly reused,
		// non-zero) still needs an explicit fill pass.
		fillChunk(chunkBuf, a.fill, elemSize)
	}

	ndcopy.Copy(chunkBuf, chunkShape, chunkSel, srcBytes, srcShape, srcSel, elemSize, nil)

	encoded, err := a.pipeline.Encode(codec.ArrayChunk{
		Shape: append([]int64(nil), chunkShape...),
		Dtype: a.meta.DataType,
		Data:  chunkBuf,
		Fill:  a.fill,
	})
	if err != nil {
		return fmt.Errorf("encode chunk %v: %w", []int64(coord), err)
	}
	if err := ws.Set(ctx, key, encoded); err != nil {
		return fmt.Errorf("write chunk %v: %w", []int64(coord), err)
	}
	return nil
}

// fillChunk replicates one element's fill bytes across buf, or zeroes buf
// when fill is the zero value. Always writes every byte explicitly — it
// does NOT assume buf arrives pre-zeroed, since callers may pass a pooled,
// reused buffer rather than a fresh make().
func fillChunk(buf, fill []byte, elemSize int) {
	if len(fill) != elemSize || elemSize == 0 || !hasNonZeroFill(fill) {
		clear(buf)
		return
	}
	for i := 0; i < len(buf); i += elemSize {
		copy(buf[i:i+elemSize], fill)
	}
}

// Stream yields decoded chunks intersecting sel, one whole chunk at a time,
// without assembling the full output buffer. Chunks are fetched and decoded
// sequentially; a missing chunk is yielded filled with the fill value.
//
// Each yielded Chunk has its own freshly allocated Data, which the iterator
// never touches again: callers may retain it without copying.
func (a *Array[T]) Stream(ctx context.Context, sel Selection) iter.Seq2[Chunk[T], error] {
	return func(yield func(Chunk[T], error) bool) {
		if err := sel.Validate(a.meta.Shape); err != nil {
			yield(Chunk[T]{}, err)
			return
		}
		chunkShape := a.meta.ChunkGrid.Configuration.ChunkShape
		elemSize := int(unsafe.Sizeof(*new(T)))
		// Scratch for edge-chunk extraction, reused across chunks.
		edgeRank := len(chunkShape)
		edgeScratch := ndcopy.NewScratch(edgeRank)
		edgeRegion := make([]ndcopy.Region, edgeRank)
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
			ck := codec.ArrayChunk{Shape: append([]int64(nil), chunkShape...), Dtype: a.meta.DataType, Fill: a.fill}
			if err := a.pipeline.Decode(raw, &ck); err != nil {
				yield(Chunk[T]{Coord: []int64(coord)}, &DecodeError{Coord: []int64(coord), Err: err})
				return
			}
			chunkBytes := elemSize
			for _, d := range chunkShape {
				chunkBytes *= int(d)
			}
			if len(ck.Data) != chunkBytes {
				yield(Chunk[T]{Coord: []int64(coord)}, &DecodeError{
					Coord: []int64(coord),
					Err:   fmt.Errorf("decoded chunk size %d != expected %d", len(ck.Data), chunkBytes),
				})
				return
			}
			var data []T
			if clipped := !slices.Equal(actualChunk, chunkShape); clipped {
				// Edge chunk: the stored chunk covers the full chunkShape, but
				// the yielded Shape is clipped to the array bounds. Extract the
				// in-bounds region into a dense buffer so Data's layout matches
				// Shape (same layout the missing-chunk branch produces).
				n := int64(1)
				for _, d := range actualChunk {
					n *= d
				}
				data = make([]T, n)
				for i := 0; i < edgeRank; i++ {
					edgeRegion[i] = ndcopy.Region{Start: 0, Stop: actualChunk[i], Step: 1}
				}
				var dataBytes []byte
				if n > 0 {
					dataBytes = unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), len(data)*elemSize)
				}
				ndcopy.Copy(dataBytes, actualChunk, edgeRegion, ck.Data, chunkShape, edgeRegion, elemSize, edgeScratch)
			} else {
				data = unsafe.Slice((*T)(unsafe.Pointer(&ck.Data[0])), len(ck.Data)/elemSize)
			}
			if !yield(Chunk[T]{Coord: []int64(coord), Shape: actualChunk, Data: data}, nil) {
				return
			}
		}
	}
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

// fillTyped writes the fill value (one element's worth of bytes in native
// byte order) across every element of dst.
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
