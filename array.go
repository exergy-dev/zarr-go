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

// pipelinePartialDecoder returns a non-nil PartialDecoder when the AB codec
// implements the optional codec.PartialDecoder interface AND no BB codecs sit
// on top of it (BB on top of sharding doesn't make sense — sharding is
// always the outermost AB). Otherwise returns nil. Result is cached on
// Array at open time so the per-chunk hot path needs no type assertion.
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
// [bytes(<host endian>)] with no AA or BB codecs. In that case Pipeline.Decode
// degenerates to copy(out, in) and can be skipped — the raw chunk bytes are
// already the native element bytes and can be fed straight into ndcopy.Copy.
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
		return bc.Endian == bytescodec.Little
	}
	return bc.Endian == bytescodec.Big
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
	// partialDecoder is non-nil when the pipeline's AB codec implements
	// codec.PartialDecoder AND no BB codecs sit on top of it. Cached here
	// (computed once at open) so per-chunk reads avoid a type assertion.
	partialDecoder codec.PartialDecoder
	// rawCopyFast is set when the codec pipeline is exactly [bytes(<host endian>)]
	// with no AA or BB codecs, so the on-wire chunk bytes can be consumed
	// directly without invoking Pipeline.Decode.
	rawCopyFast bool
}

// Chunk is one decoded chunk yielded from Stream. Data aliases an internal
// buffer owned by the iterator and is only valid for the current loop-body
// iteration; see Stream's doc comment for the full lifetime contract.
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

// maxStackRank is the maximum array rank for which per-chunk scratch buffers
// are kept on the stack (as fixed-size arrays) instead of heap-allocated
// per-chunk slices. Real-world Zarr arrays are almost always rank <= 8;
// higher ranks fall back to heap allocation, which is correct but slower.
const maxStackRank = 8

// chunkScratch holds fixed-size scratch arrays used per chunk during Read
// and Write. When rank <= maxStackRank, slicing these arrays avoids heap
// allocation because the whole struct lives on the calling goroutine's
// stack. chunkSrc/outDst are filled by chunkSelectionInto (in the zarr
// package's own DimRange type, the public selection-description type);
// srcRegion/dstRegion hold the same ranges converted to ndcopy.Region (the
// shared copy kernel's independent type — see toRegions) for feeding
// ndcopy.Copy. srcStrides/dstStrides/regionLen/idx back an ndcopy.Scratch.
type chunkScratch struct {
	chunkSrc   [maxStackRank]DimRange
	outDst     [maxStackRank]DimRange
	srcRegion  [maxStackRank]ndcopy.Region
	dstRegion  [maxStackRank]ndcopy.Region
	srcStrides [maxStackRank]int64
	dstStrides [maxStackRank]int64
	regionLen  [maxStackRank]int64
	idx        [maxStackRank]int64
}

// toRegions converts DimRange values (zarr's public selection-description
// type) into ndcopy.Region values (the shared copy kernel's independent
// type — kept separate so the internal copy-kernel package doesn't dictate
// the public API's type). The two have identical field layout but Go does
// not allow a direct slice conversion between differently-named struct
// types, hence this explicit per-element copy.
func toRegions(dst []ndcopy.Region, src []DimRange) {
	for i, r := range src {
		dst[i] = ndcopy.Region{Start: r.Start, Stop: r.Stop, Step: r.Step}
	}
}

// newChunkScratch returns rank-sized working slices for one chunk's
// selection-to-region conversion, backed by scratch's stack-allocated
// arrays when rank fits (the common case), or heap-allocated otherwise.
// Shared by partialDecodeAndCopy, decodeAndCopy, and writeOneChunk.
func newChunkScratch(scratch *chunkScratch, rank int) (srcSlice, dstSlice []DimRange, srcRegion, dstRegion []ndcopy.Region, ndSc ndcopy.Scratch) {
	if rank <= maxStackRank {
		srcSlice = scratch.chunkSrc[:rank]
		dstSlice = scratch.outDst[:rank]
		srcRegion = scratch.srcRegion[:rank]
		dstRegion = scratch.dstRegion[:rank]
		ndSc = ndcopy.Scratch{
			SrcStrides: scratch.srcStrides[:rank],
			DstStrides: scratch.dstStrides[:rank],
			RegionLen:  scratch.regionLen[:rank],
			Idx:        scratch.idx[:rank],
		}
		return
	}
	srcSlice = make([]DimRange, rank)
	dstSlice = make([]DimRange, rank)
	srcRegion = make([]ndcopy.Region, rank)
	dstRegion = make([]ndcopy.Region, rank)
	ndSc = *ndcopy.NewScratch(rank)
	return
}

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

	// Collect chunk coords. For rank <= maxStackRank, store coords as
	// fixed-size arrays to avoid one []int64 allocation per chunk
	// (ChunkCoord.Clone). For higher ranks, fall back to heap-allocated
	// ChunkCoord clones.
	var workFixed [][maxStackRank]int64
	var workHeap []ChunkCoord
	if rank <= maxStackRank {
		for c := range ChunksIntersecting(a.meta.Shape, chunkShape, sel) {
			var fc [maxStackRank]int64
			copy(fc[:rank], c)
			workFixed = append(workFixed, fc)
		}
	} else {
		for c := range ChunksIntersecting(a.meta.Shape, chunkShape, sel) {
			workHeap = append(workHeap, c.Clone())
		}
	}
	nwork := len(workFixed) + len(workHeap)
	if nwork == 0 {
		return nil
	}

	// ioSem bounds in-flight store fetches (and, for the partial-decode path,
	// the whole fetch+inner-decode operation, which is I/O-bound overall).
	// decodeSem separately bounds the CPU-bound full-chunk decode step, so a
	// high ioConcurrency (good for a network-bound store) doesn't also
	// oversubscribe CPU decode workers.
	ioConc := a.opts.ioConcurrency
	if ioConc > nwork {
		ioConc = nwork
	}
	if ioConc < 1 {
		ioConc = 1
	}
	decodeConc := a.opts.concurrency
	if decodeConc > nwork {
		decodeConc = nwork
	}
	if decodeConc < 1 {
		decodeConc = 1
	}
	ioSem := make(chan struct{}, ioConc)
	decodeSem := make(chan struct{}, decodeConc)
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

	launch := func(coord ChunkCoord) {
		select {
		case <-cctx.Done():
			return
		case ioSem <- struct{}{}:
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-ioSem }()
			key := metadata.ChunkKey(a.path, a.meta.ChunkKeyEnc, []int64(coord))
			if a.partialDecoder != nil {
				// Interleaves I/O and per-inner-chunk decode internally;
				// runs entirely under the I/O gate, no decodeSem.
				if err := a.partialDecodeAndCopy(cctx, a.partialDecoder, coord, key, sel, chunkShape, outShape, dst); err != nil {
					if errors.Is(err, store.ErrKeyNotFound) {
						return
					}
					setErr(err)
				}
				return
			}
			var (
				raw     []byte
				release func()
				err     error
			)
			if a.mmap != nil {
				raw, release, err = a.mmap.GetMmap(cctx, key)
			} else {
				raw, err = a.store.Get(cctx, key)
			}
			if err != nil {
				if errors.Is(err, store.ErrKeyNotFound) {
					return // missing chunk = fill, already in dst
				}
				setErr(fmt.Errorf("fetch chunk %v: %w", []int64(coord), err))
				return
			}
			if release != nil {
				defer release()
			}
			select {
			case <-cctx.Done():
				return
			case decodeSem <- struct{}{}:
			}
			err = a.decodeAndCopy(coord, raw, sel, chunkShape, outShape, dst)
			<-decodeSem
			if err != nil {
				setErr(err)
			}
		}()
	}

	for i := range workFixed {
		// &workFixed[i] is stable for the lifetime of this Read call;
		// the goroutine retains the slice header but the underlying
		// array lives in workFixed's backing storage.
		launch(ChunkCoord(workFixed[i][:rank]))
	}
	for _, c := range workHeap {
		launch(c)
	}
	wg.Wait()
	if firstErr == nil {
		// If ctx was cancelled, launch() may have skipped chunks without any
		// goroutine recording an error; returning nil would silently hand
		// back incomplete data.
		firstErr = context.Cause(ctx)
	}
	return firstErr
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
// If the per-chunk source range degenerates to "the entire (clipped) chunk"
// or any dim has step != 1, this falls back to the full-fetch path so the
// pipeline.Decode path is still exercised correctly.
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
	var scratch chunkScratch
	srcSlice, dstSlice, srcRegion, dstRegion, ndSc := newChunkScratch(&scratch, rank)
	chunkSelectionInto(a.meta.Shape, chunkShape, sel, coord, srcSlice, dstSlice)

	// Partial decode requires step==1; bail to the full path if any dim is stepped.
	actual := make([]int64, rank)
	a.chunkActualShape(coord, chunkShape, actual)
	isFull := true
	for d := 0; d < rank; d++ {
		if srcSlice[d].Step != 1 {
			// Stepped: full fetch + decode handles striding correctly.
			raw, err := a.store.Get(ctx, key)
			if err != nil {
				return err
			}
			return a.decodeAndCopy(coord, raw, sel, chunkShape, outShape, dst)
		}
		if srcSlice[d].Start != 0 || srcSlice[d].Stop != actual[d] {
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
	region := make([]codec.DimRange, rank)
	regionElems := int64(1)
	for d := 0; d < rank; d++ {
		region[d] = codec.DimRange{Start: srcSlice[d].Start, Stop: srcSlice[d].Stop, Step: 1}
		regionShape[d] = srcSlice[d].Stop - srcSlice[d].Start
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
		Region:    region,
		Out:       out,
		Fill:      a.fill,
		GapBytes:  a.opts.rangeGap,
	}
	if err := pd.DecodePartial(ctx, a.store, &req); err != nil {
		// Translate "key not found" to the same sentinel as the full path
		// so the caller skips the chunk and leaves the fill value in place.
		if errors.Is(err, store.ErrKeyNotFound) {
			return err
		}
		return &DecodeError{Coord: []int64(coord), Codec: a.pipeline.AB.Name(), Err: err}
	}

	// Copy the dense region buffer into dst at the dstSlice positions.
	// The region buffer has shape regionShape (in chunk-local source coords)
	// and maps 1:1 onto dstSlice (Step==1 along every dim), so it can be
	// treated as if it were the whole "chunk" with a full [0, regionShape[d])
	// source region.
	for d := 0; d < rank; d++ {
		srcRegion[d] = ndcopy.Region{Start: 0, Stop: regionShape[d], Step: 1}
	}
	toRegions(dstRegion, dstSlice)
	var dstBytes []byte
	if len(dst) > 0 {
		dstBytes = unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*elemSize)
	}
	ndcopy.Copy(dstBytes, outShape, dstRegion, out, regionShape, srcRegion, elemSize, &ndSc)
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
	rank := len(chunkShape)
	var scratch chunkScratch
	srcSlice, dstSlice, srcRegion, dstRegion, ndSc := newChunkScratch(&scratch, rank)
	chunkSelectionInto(a.meta.Shape, chunkShape, sel, coord, srcSlice, dstSlice)
	toRegions(srcRegion, srcSlice)
	toRegions(dstRegion, dstSlice)
	elemSize := int(unsafe.Sizeof(*new(T)))
	chunkBytes := elemSize
	for _, d := range chunkShape {
		chunkBytes *= int(d)
	}
	var dstBytes []byte
	if len(dst) > 0 {
		dstBytes = unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*elemSize)
	}

	// Fast path: pipeline is [bytes(<host endian>)], no AA/BB. Raw bytes
	// are already native element bytes — skip pipeline.Decode and the
	// pooled buffer entirely.
	if a.rawCopyFast && len(raw) == chunkBytes {
		ndcopy.Copy(dstBytes, outShape, dstRegion, raw, chunkShape, srcRegion, elemSize, &ndSc)
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
		return &DecodeError{Coord: []int64(coord), Codec: a.pipeline.AB.Name(), Err: err}
	}
	if len(chunk.Data) != chunkBytes {
		return &DecodeError{Coord: []int64(coord), Codec: a.pipeline.AB.Name(),
			Err: fmt.Errorf("decoded chunk size %d != expected %d", len(chunk.Data), chunkBytes)}
	}

	ndcopy.Copy(dstBytes, outShape, dstRegion, chunk.Data, chunkShape, srcRegion, elemSize, &ndSc)
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
	ws store.WritableStore,
	coord ChunkCoord,
	sel Selection,
	chunkShape, srcShape []int64,
	srcBytes []byte,
	elemSize int,
) error {
	rank := len(chunkShape)
	var scratch chunkScratch
	// dstRegion is the chunk side (write destination; may be stepped),
	// srcRegion is the user src side (read source; always step1) — see
	// newChunkScratch's doc for how these map onto the shared scratch.
	chunkSrcSlice, outDstSlice, srcRegion, dstRegion, ndSc := newChunkScratch(&scratch, rank)
	chunkSelectionInto(a.meta.Shape, chunkShape, sel, coord, chunkSrcSlice, outDstSlice)

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
		if chunkSrcSlice[i].Start != 0 || chunkSrcSlice[i].Stop != inArrayExtent || chunkSrcSlice[i].Step != 1 {
			fullCovered = false
		}
		if inArrayExtent != chunkShape[i] {
			noPadding = false
		}
	}

	key := metadata.ChunkKey(a.path, a.meta.ChunkKeyEnc, []int64(coord))

	if !fullCovered {
		// Read-modify-write: try to load existing chunk.
		existing, err := a.store.Get(ctx, key)
		switch {
		case err == nil:
			// Decode existing into chunkBuf.
			ck := codec.ArrayChunk{Shape: append([]int64(nil), chunkShape...), Dtype: a.meta.DataType, Data: chunkBuf[:0], Fill: a.fill}
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
	} else if !noPadding || hasNonZeroFill(a.fill) {
		// Either padded edge positions need fill, or every in-array byte is
		// about to be overwritten but chunkBuf (pooled — possibly reused,
		// non-zero) still needs an explicit fill pass.
		fillChunk(chunkBuf, a.fill, elemSize)
	}

	// Copy from src into chunkBuf: the chunk side (chunkSrcSlice) is the
	// copy destination and may be stepped; the user-src side (outDstSlice)
	// is the copy source and is always step1.
	toRegions(dstRegion, chunkSrcSlice)
	toRegions(srcRegion, outDstSlice)
	ndcopy.Copy(chunkBuf, chunkShape, dstRegion, srcBytes, srcShape, srcRegion, elemSize, &ndSc)

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

// Stream yields decoded chunks intersecting sel without assembling the full
// output buffer.
//
// The Data slice in each yielded Chunk[T] aliases an internal decode buffer
// owned by the iterator. It is valid for the lifetime of the body of the
// for-range loop only; callers that need to retain the bytes across
// iterations must copy the slice. The iterator allocates a fresh buffer per
// chunk so consecutive yields don't overlap, but reuse may begin once the
// loop body returns true.
func (a *Array[T]) Stream(ctx context.Context, sel Selection) iter.Seq2[Chunk[T], error] {
	return func(yield func(Chunk[T], error) bool) {
		if err := sel.Validate(a.meta.Shape); err != nil {
			yield(Chunk[T]{}, err)
			return
		}
		chunkShape := a.meta.ChunkGrid.Configuration.ChunkShape
		elemSize := int(unsafe.Sizeof(*new(T)))
		// Scratch for the edge-chunk extraction branch below. chunkShape
		// (hence rank) is constant for the whole Stream call, so these are
		// allocated once and reused across every edge chunk instead of
		// per-chunk.
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
			ck := codec.ArrayChunk{Shape: append([]int64(nil), chunkShape...), Dtype: a.meta.DataType, Fill: a.fill}
			if err := a.pipeline.Decode(raw, &ck); err != nil {
				yield(Chunk[T]{Coord: []int64(coord)}, &DecodeError{
					Coord: []int64(coord), Codec: a.pipeline.AB.Name(), Err: err,
				})
				return
			}
			chunkBytes := elemSize
			for _, d := range chunkShape {
				chunkBytes *= int(d)
			}
			if len(ck.Data) != chunkBytes {
				yield(Chunk[T]{Coord: []int64(coord)}, &DecodeError{
					Coord: []int64(coord), Codec: a.pipeline.AB.Name(),
					Err: fmt.Errorf("decoded chunk size %d != expected %d", len(ck.Data), chunkBytes),
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
