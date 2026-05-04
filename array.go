package zarr

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"sync"
	"unsafe"

	"github.com/rkm/zarr-go/codec"
	bytescodec "github.com/rkm/zarr-go/codec/bytes"
	"github.com/rkm/zarr-go/internal/bufpool"
	"github.com/rkm/zarr-go/internal/metadata"
	"github.com/rkm/zarr-go/store"
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
func pipelinePartialDecoder(p *codec.Pipeline) (codec.PartialDecoder, bool) {
	if p == nil || p.AB == nil || len(p.BB) != 0 {
		return nil, false
	}
	pd, ok := p.AB.(codec.PartialDecoder)
	if !ok {
		return nil, false
	}
	return pd, true
}

// isRawCopyFastPipeline reports whether the pipeline is exactly
// [bytes(<host endian>)] with no AA or BB codecs. In that case Pipeline.Decode
// degenerates to copy(out, in) and can be skipped — the raw chunk bytes are
// already the native element bytes and can be fed straight into copyChunkRegion.
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
	fill     []byte // fill value bytes, native LE, sizeof(T)
	opts     arrayOptions
	// partialDecoder is non-nil when the pipeline's AB codec implements
	// codec.PartialDecoder AND no BB codecs sit on top of it. Cached here
	// (computed once at open) so per-chunk reads avoid a type assertion.
	// hasPartial mirrors `partialDecoder != nil` as a single byte; the hot
	// path checks this byte instead of the 2-word interface compare.
	partialDecoder codec.PartialDecoder
	hasPartial     bool
	// rawCopyFast is set when the codec pipeline is exactly [bytes(<host endian>)]
	// with no AA or BB codecs, so the on-wire chunk bytes can be consumed
	// directly without invoking Pipeline.Decode.
	rawCopyFast bool
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

// maxStackRank is the maximum array rank for which per-chunk scratch buffers
// are kept on the stack (as fixed-size arrays) instead of heap-allocated
// per-chunk slices. Real-world Zarr arrays are almost always rank <= 8;
// higher ranks fall back to heap allocation, which is correct but slower.
const maxStackRank = 8

// chunkScratch holds fixed-size scratch arrays used per chunk during Read.
// When rank <= maxStackRank, slicing these arrays avoids heap allocation
// because the whole struct lives on the calling goroutine's stack.
type chunkScratch struct {
	chunkSrc     [maxStackRank]DimRange
	outDst       [maxStackRank]DimRange
	chunkStrides [maxStackRank]int64
	regionLen    [maxStackRank]int64
	idx          [maxStackRank]int64
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
	outStrides := rowMajorStrides(outShape)

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

	conc := a.opts.concurrency
	if conc > nwork {
		conc = nwork
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

	launch := func(coord ChunkCoord) {
		select {
		case <-cctx.Done():
			return
		case sem <- struct{}{}:
		}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			key := metadata.ChunkKey(a.path, a.meta.ChunkKeyEnc, []int64(coord))
			if a.hasPartial {
				if err := a.partialDecodeAndCopy(cctx, a.partialDecoder, coord, key, sel, chunkShape, outShape, outStrides, dst); err != nil {
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
			if err := a.decodeAndCopy(coord, raw, sel, chunkShape, outShape, outStrides, dst); err != nil {
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
	if firstErr != nil {
		return firstErr
	}
	return nil
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
	chunkShape, outShape, outStrides []int64,
	dst []T,
) error {
	rank := len(chunkShape)
	var scratch chunkScratch
	var (
		srcSlice, dstSlice                     []DimRange
		stridesSlice, regionLenSlice, idxSlice []int64
	)
	if rank <= maxStackRank {
		srcSlice = scratch.chunkSrc[:rank]
		dstSlice = scratch.outDst[:rank]
		stridesSlice = scratch.chunkStrides[:rank]
		regionLenSlice = scratch.regionLen[:rank]
		idxSlice = scratch.idx[:rank]
	} else {
		srcSlice = make([]DimRange, rank)
		dstSlice = make([]DimRange, rank)
		stridesSlice = make([]int64, rank)
		regionLenSlice = make([]int64, rank)
		idxSlice = make([]int64, rank)
	}
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
			return a.decodeAndCopy(coord, raw, sel, chunkShape, outShape, outStrides, dst)
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
		return a.decodeAndCopy(coord, raw, sel, chunkShape, outShape, outStrides, dst)
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

	// Use the AB codec's DecodePartial; thread the store as the fetcher.
	fetcher := newStoreFetcher(a.store)
	if err := pd.DecodePartial(ctx, fetcher, key, actual, a.meta.DataType, region, out, a.fill); err != nil {
		// Translate "key not found" to the same sentinel as the full path
		// so the caller skips the chunk and leaves the fill value in place.
		if errors.Is(err, store.ErrKeyNotFound) {
			return err
		}
		return &DecodeError{Coord: []int64(coord), Codec: a.pipeline.AB.Name(), Err: err}
	}

	// Copy the dense region buffer into dst at the dstSlice positions.
	// The region buffer has shape regionShape (in chunk-local source coords),
	// dst has shape outShape with strides outStrides, and the region maps
	// 1:1 onto dstSlice (Step==1 along every dim).
	//
	// We can express this via copyChunkRegion by treating the region buffer
	// as if it were the chunk and using DimRange{0, regionShape[d], 1} for
	// chunkSrc.
	srcAll := scratch.chunkSrc[:rank]
	for d := 0; d < rank; d++ {
		srcAll[d] = DimRange{Start: 0, Stop: regionShape[d], Step: 1}
	}
	copyChunkRegion[T](
		out, regionShape, srcAll,
		dst, outShape, outStrides, dstSlice,
		elemSize,
		stridesSlice, regionLenSlice, idxSlice,
	)
	return nil
}

// storeFetcher adapts a store.Store to codec.RangeFetcher (and, when the
// underlying store has the methods, codec.sharding.SuffixRangeFetcher). The
// adapter caches the blob length on first GetRange so subsequent suffix
// requests can be served by a single positional GetRange — important for
// backends that don't expose a native suffix-range API.
type storeFetcher struct {
	s   store.Store
	mu  sync.Mutex
	len map[string]int64
}

func newStoreFetcher(s store.Store) *storeFetcher {
	return &storeFetcher{s: s, len: make(map[string]int64)}
}

// GetRange forwards to the underlying store's GetRange.
func (f *storeFetcher) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	return f.s.GetRange(ctx, key, off, n)
}

// GetSuffix returns the last n bytes of the value at key. It first probes
// the store for the blob's length (via a Get followed by len(); cached for
// the lifetime of this fetcher), then issues a positional GetRange.
//
// For backends that natively support HTTP `Range: bytes=-N`, a future
// optimization can short-circuit this by detecting the backend and issuing
// the suffix directly. For now, the cache means we pay one whole-blob fetch
// per shard (which the codec then mostly re-uses via slicedFetcher fallback).
//
// NOTE: We intentionally fall back to whole-blob Get here because none of
// the in-tree backends expose blob length cheaply; for cloud workloads the
// dominant codepath is one suffix-range HTTP request. Backends should add a
// real GetSuffix on top of this adapter to enable that path.
func (f *storeFetcher) GetSuffix(ctx context.Context, key string, n int64) ([]byte, error) {
	f.mu.Lock()
	knownLen, ok := f.len[key]
	f.mu.Unlock()
	if ok {
		off := knownLen - n
		if off < 0 {
			off = 0
			n = knownLen
		}
		return f.s.GetRange(ctx, key, off, n)
	}
	// First call for this key: fetch the whole blob to learn its length and
	// return the suffix from it. Cache the length for future calls.
	whole, err := f.s.Get(ctx, key)
	if err != nil {
		return nil, err
	}
	f.mu.Lock()
	f.len[key] = int64(len(whole))
	f.mu.Unlock()
	if int64(len(whole)) < n {
		return whole, nil
	}
	return whole[int64(len(whole))-n:], nil
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
	rank := len(chunkShape)
	var scratch chunkScratch
	var (
		srcSlice, dstSlice                     []DimRange
		stridesSlice, regionLenSlice, idxSlice []int64
	)
	if rank <= maxStackRank {
		srcSlice = scratch.chunkSrc[:rank]
		dstSlice = scratch.outDst[:rank]
		stridesSlice = scratch.chunkStrides[:rank]
		regionLenSlice = scratch.regionLen[:rank]
		idxSlice = scratch.idx[:rank]
	} else {
		srcSlice = make([]DimRange, rank)
		dstSlice = make([]DimRange, rank)
		stridesSlice = make([]int64, rank)
		regionLenSlice = make([]int64, rank)
		idxSlice = make([]int64, rank)
	}
	chunkSelectionInto(a.meta.Shape, chunkShape, sel, coord, srcSlice, dstSlice)
	elemSize := int(unsafe.Sizeof(*new(T)))
	chunkBytes := elemSize
	for _, d := range chunkShape {
		chunkBytes *= int(d)
	}

	// Fast path: pipeline is [bytes(<host endian>)], no AA/BB. Raw bytes
	// are already native element bytes — skip pipeline.Decode and the
	// pooled buffer entirely.
	if a.rawCopyFast && len(raw) == chunkBytes {
		copyChunkRegion[T](
			raw, chunkShape, srcSlice,
			dst, outShape, outStrides, dstSlice,
			elemSize,
			stridesSlice, regionLenSlice, idxSlice,
		)
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

	copyChunkRegion[T](
		chunk.Data, chunkShape, srcSlice,
		dst, outShape, outStrides, dstSlice,
		elemSize,
		stridesSlice, regionLenSlice, idxSlice,
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
	s := make([]int64, len(shape))
	rowMajorStridesInto(shape, s)
	return s
}

// rowMajorStridesInto fills out (len rank) with the row-major element strides
// for shape. Caller-allocated variant of rowMajorStrides.
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
//
// Fast path: when all chunkSrc steps are 1 we collapse the inner contiguous
// dims into a single memmove per outer-iteration position. The maximum number
// of trailing dims we can merge is bounded by how many inner dims span the
// full chunk row on the source side AND the full output row on the dest side
// (the outermost merged dim itself may be a partial range).
func copyChunkRegion[T Numeric](
	chunkBytes []byte, chunkShape []int64, chunkSrc []DimRange,
	dst []T, outShape, outStrides []int64, outDst []DimRange,
	elemSize int,
	chunkStrides, regionLen, idx []int64,
) {
	rank := len(chunkShape)
	rowMajorStridesInto(chunkShape, chunkStrides)
	for i := 0; i < rank; i++ {
		regionLen[i] = outDst[i].Stop - outDst[i].Start
		idx[i] = 0
	}
	total := int64(1)
	for _, n := range regionLen {
		total *= n
	}
	if total == 0 {
		return
	}
	dstBytes := unsafe.Slice((*byte)(unsafe.Pointer(&dst[0])), len(dst)*elemSize)

	if rank == 0 {
		return
	}

	// Detect the all-step==1 case (the dominant Read pattern).
	allStep1 := true
	for d := 0; d < rank; d++ {
		if chunkSrc[d].Step != 1 {
			allStep1 = false
			break
		}
	}

	if allStep1 {
		// Determine how many trailing dims we can collapse into a single
		// contiguous copy. Dim d (with d > mergeStart) must span the FULL
		// chunk dim on src AND the FULL output dim on dst — otherwise the
		// destination memory is broken up by a stride and we can't merge
		// across it. The outermost merged dim (mergeStart) may be partial.
		mergeStart := rank - 1
		for mergeStart > 0 {
			d := mergeStart // candidate inner dim already in the merged block
			if !(chunkSrc[d].Start == 0 && chunkSrc[d].Stop == chunkShape[d] &&
				outDst[d].Start == 0 && outDst[d].Stop == outShape[d]) {
				break
			}
			mergeStart--
		}

		// Per-call element count for the merged inner block.
		innerElems := int64(1)
		for d := mergeStart; d < rank; d++ {
			innerElems *= regionLen[d]
		}
		innerBytes := innerElems * int64(elemSize)

		// Outer dims iterate one row at a time.
		outerRank := mergeStart
		if outerRank == 0 {
			// The whole region collapses to one copy.
			var srcEl, dstEl int64
			for d := 0; d < rank; d++ {
				srcEl += chunkSrc[d].Start * chunkStrides[d]
				dstEl += outDst[d].Start * outStrides[d]
			}
			srcOff := srcEl * int64(elemSize)
			dstOff := dstEl * int64(elemSize)
			copy(dstBytes[dstOff:dstOff+innerBytes], chunkBytes[srcOff:srcOff+innerBytes])
			return
		}

		// Pre-compute the constant base offsets contributed by the inner
		// (merged) dims; each inner merged dim has Start either 0 (full
		// span) or some value (only the outermost merged dim can be
		// partial). Both contributions are constant across iterations.
		var innerSrcBase, innerDstBase int64
		for d := mergeStart; d < rank; d++ {
			innerSrcBase += chunkSrc[d].Start * chunkStrides[d]
			innerDstBase += outDst[d].Start * outStrides[d]
		}

		// Reuse caller-supplied idx (sliced to outerRank).
		outerIdx := idx[:outerRank]
		for d := 0; d < outerRank; d++ {
			outerIdx[d] = 0
		}
		// Total iterations over the outer dims.
		outerTotal := int64(1)
		for d := 0; d < outerRank; d++ {
			outerTotal *= regionLen[d]
		}
		for k := int64(0); k < outerTotal; k++ {
			var srcEl, dstEl int64
			for d := 0; d < outerRank; d++ {
				srcEl += (chunkSrc[d].Start + outerIdx[d]) * chunkStrides[d]
				dstEl += (outDst[d].Start + outerIdx[d]) * outStrides[d]
			}
			srcOff := (srcEl + innerSrcBase) * int64(elemSize)
			dstOff := (dstEl + innerDstBase) * int64(elemSize)
			copy(dstBytes[dstOff:dstOff+innerBytes], chunkBytes[srcOff:srcOff+innerBytes])
			// Increment outerIdx row-major (last outer dim fastest).
			for d := outerRank - 1; d >= 0; d-- {
				outerIdx[d]++
				if outerIdx[d] < regionLen[d] {
					break
				}
				outerIdx[d] = 0
			}
		}
		return
	}

	// Fallback: stride > 1 in some dim — element-by-element copy.
	// idx already zeroed at function entry.
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
