package zarr_test

import (
	"context"
	"encoding/json"
	"iter"
	"sync/atomic"
	"testing"
	"time"

	"github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/codec"
	"github.com/exergy-dev/zarr-go/store"
	"github.com/exergy-dev/zarr-go/store/mem"
)

const shardCfg2x2 = `{
  "chunk_shape": [2, 2],
  "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
  "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}]
}`

// countingStore wraps a store.Store, counting Get/GetRange/GetSuffix calls
// so tests can assert the partial-decode path issues fewer/cheaper requests
// than a full fetch would.
type countingStore struct {
	inner                        store.Store
	gets, getRanges, getSuffixes atomic.Int64
}

func (c *countingStore) Get(ctx context.Context, key string) ([]byte, error) {
	c.gets.Add(1)
	return c.inner.Get(ctx, key)
}

func (c *countingStore) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	c.getRanges.Add(1)
	return c.inner.GetRange(ctx, key, off, n)
}

func (c *countingStore) GetSuffix(ctx context.Context, key string, n int64) ([]byte, error) {
	c.getSuffixes.Add(1)
	return c.inner.(store.SuffixReader).GetSuffix(ctx, key, n)
}

func (c *countingStore) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	return c.inner.List(ctx, prefix)
}

func (c *countingStore) Exists(ctx context.Context, key string) (bool, error) {
	return c.inner.Exists(ctx, key)
}

var (
	_ store.Store        = (*countingStore)(nil)
	_ store.SuffixReader = (*countingStore)(nil)
)

// TestPartialRead_EdgeShard proves the fullShape contract fix: a sharded
// array whose shape is not divisible by its (outer) chunk shape has an edge
// chunk whose array-bounds-clipped extent is smaller than the shard's FULL
// stored shape. A genuinely-partial (non-isFull) read into that edge chunk
// must still resolve correctly — pre-fix, DecodePartial was fed the clipped
// shape as fullShape, which computes the wrong inner-chunk grid against a
// shard that was actually encoded (and its index built) at the full nominal
// chunk_shape.
func TestPartialRead_EdgeShard(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	shape := []int64{10, 10}
	chunks := []int64{8, 8} // outer chunk (1,1) clips to actual [2,2] within a full [8,8] shard
	a, err := zarr.CreateArray[int32](ctx, s, "arr", shape, chunks,
		zarr.WithCodecs(codec.Spec{Name: "sharding_indexed", Configuration: json.RawMessage(shardCfg2x2)}))
	if err != nil {
		t.Fatalf("CreateArray: %v", err)
	}
	src := make([]int32, 10*10)
	for i := range src {
		src[i] = int32(i + 1)
	}
	if err := a.Write(ctx, zarr.All(shape), src); err != nil {
		t.Fatalf("Write: %v", err)
	}

	b, err := zarr.OpenArray[int32](ctx, s, "arr")
	if err != nil {
		t.Fatalf("OpenArray: %v", err)
	}
	ref := make([]int32, 10*10)
	if err := b.Read(ctx, zarr.All(shape), ref); err != nil {
		t.Fatalf("ref Read: %v", err)
	}

	// Genuine partial sub-region of the edge chunk (1,1): only row 8, cols
	// 8-9 out of its full clipped extent rows[8,10) cols[8,10) — isFull must
	// be false so this actually goes through DecodePartial.
	sel := zarr.Selection{Ranges: []zarr.DimRange{
		{Start: 8, Stop: 9, Step: 1},
		{Start: 8, Stop: 10, Step: 1},
	}}
	got := make([]int32, sel.NumElements())
	if err := b.Read(ctx, sel, got); err != nil {
		t.Fatalf("edge-shard partial Read: %v", err)
	}
	want := []int32{ref[8*10+8], ref[8*10+9]}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

// TestPartialRead_Rank9 exercises the heap-allocated (rank > maxStackRank)
// scratch path in partialDecodeAndCopy with a genuinely-partial selection,
// guarding against a latent panic in that path.
func TestPartialRead_Rank9(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	const rank = 9
	shape := make([]int64, rank)
	chunks := make([]int64, rank)
	innerChunkShape := make([]int64, rank)
	for i := range shape {
		shape[i] = 2
		chunks[i] = 2
		innerChunkShape[i] = 1
	}
	cfg, err := json.Marshal(map[string]any{
		"chunk_shape": innerChunkShape,
		"codecs":      []map[string]any{{"name": "bytes", "configuration": map[string]any{"endian": "little"}}},
		"index_codecs": []map[string]any{
			{"name": "bytes", "configuration": map[string]any{"endian": "little"}},
			{"name": "crc32c"},
		},
	})
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	a, err := zarr.CreateArray[int32](ctx, s, "arr", shape, chunks,
		zarr.WithCodecs(codec.Spec{Name: "sharding_indexed", Configuration: cfg}))
	if err != nil {
		t.Fatalf("CreateArray: %v", err)
	}

	total := int64(1)
	for _, d := range shape {
		total *= d
	}
	src := make([]int32, total)
	for i := range src {
		src[i] = int32(i + 1)
	}
	if err := a.Write(ctx, zarr.All(shape), src); err != nil {
		t.Fatalf("Write: %v", err)
	}

	b, err := zarr.OpenArray[int32](ctx, s, "arr")
	if err != nil {
		t.Fatalf("OpenArray: %v", err)
	}
	ref := make([]int32, total)
	if err := b.Read(ctx, zarr.All(shape), ref); err != nil {
		t.Fatalf("ref Read: %v", err)
	}

	// Full range on every dim except the last, restricted to [0,1) out of
	// [0,2) — a genuine partial selection.
	ranges := make([]zarr.DimRange, rank)
	for i := 0; i < rank; i++ {
		ranges[i] = zarr.DimRange{Start: 0, Stop: shape[i], Step: 1}
	}
	ranges[rank-1] = zarr.DimRange{Start: 0, Stop: 1, Step: 1}
	sel := zarr.Selection{Ranges: ranges}
	got := make([]int32, sel.NumElements())
	if err := b.Read(ctx, sel, got); err != nil {
		t.Fatalf("rank-9 partial Read: %v", err)
	}

	idx := 0
	coord := make([]int64, rank)
	var walk func(d int)
	walk = func(d int) {
		if d == rank {
			flat := int64(0)
			for i, c := range coord {
				flat = flat*shape[i] + c
			}
			if got[idx] != ref[flat] {
				t.Fatalf("coord %v: got %d want %d", append([]int64(nil), coord...), got[idx], ref[flat])
			}
			idx++
			return
		}
		for v := ranges[d].Start; v < ranges[d].Stop; v++ {
			coord[d] = v
			walk(d + 1)
		}
	}
	walk(0)
}

// TestPartialRead_MissingShard verifies a genuinely-partial read into a
// never-written outer chunk resolves to the fill value with no error.
func TestPartialRead_MissingShard(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	shape := []int64{16, 16}
	chunks := []int64{8, 8}
	a, err := zarr.CreateArray[int32](ctx, s, "arr", shape, chunks,
		zarr.WithCodecs(codec.Spec{Name: "sharding_indexed", Configuration: json.RawMessage(shardCfg2x2)}))
	if err != nil {
		t.Fatalf("CreateArray: %v", err)
	}
	// Write only the top-left shard; the bottom-right shard is never touched.
	sub := zarr.Selection{Ranges: []zarr.DimRange{{Start: 0, Stop: 8, Step: 1}, {Start: 0, Stop: 8, Step: 1}}}
	src := make([]int32, sub.NumElements())
	for i := range src {
		src[i] = int32(i + 1)
	}
	if err := a.Write(ctx, sub, src); err != nil {
		t.Fatalf("Write: %v", err)
	}

	b, err := zarr.OpenArray[int32](ctx, s, "arr")
	if err != nil {
		t.Fatalf("OpenArray: %v", err)
	}
	// Genuine partial sub-region entirely inside the never-written shard.
	sel := zarr.Selection{Ranges: []zarr.DimRange{{Start: 10, Stop: 14, Step: 1}, {Start: 10, Stop: 14, Step: 1}}}
	got := make([]int32, sel.NumElements())
	if err := b.Read(ctx, sel, got); err != nil {
		t.Fatalf("missing-shard partial Read: %v", err)
	}
	for i, v := range got {
		if v != 0 {
			t.Fatalf("got[%d]=%d, want 0 (fill) for missing shard", i, v)
		}
	}
}

// TestPartialRead_FetchCounts proves the engine-level wiring end-to-end: a
// genuinely-partial read against a sharded array (mem.Store, which
// implements store.SuffixReader) issues exactly one GetSuffix (the index
// trailer) plus a small, coalesced number of GetRange calls — never a
// whole-chunk Get — matching the README's "fewer requests than a full read"
// claim.
func TestPartialRead_FetchCounts(t *testing.T) {
	ctx := context.Background()
	inner := mem.New()
	shape := []int64{16, 16}
	chunks := []int64{8, 8}
	a, err := zarr.CreateArray[int32](ctx, inner, "arr", shape, chunks,
		zarr.WithCodecs(codec.Spec{Name: "sharding_indexed", Configuration: json.RawMessage(shardCfg2x2)}))
	if err != nil {
		t.Fatalf("CreateArray: %v", err)
	}
	src := make([]int32, 16*16)
	for i := range src {
		src[i] = int32(i + 1)
	}
	if err := a.Write(ctx, zarr.All(shape), src); err != nil {
		t.Fatalf("Write: %v", err)
	}

	cs := &countingStore{inner: inner}
	b, err := zarr.OpenArray[int32](ctx, cs, "arr")
	if err != nil {
		t.Fatalf("OpenArray: %v", err)
	}
	// OpenArray itself issues a Get for zarr.json; reset counts so only the
	// Read below is measured.
	cs.gets.Store(0)
	cs.getRanges.Store(0)
	cs.getSuffixes.Store(0)
	refArr, err := zarr.OpenArray[int32](ctx, inner, "arr")
	if err != nil {
		t.Fatalf("OpenArray (ref): %v", err)
	}
	ref := make([]int32, 16*16)
	if err := refArr.Read(ctx, zarr.All(shape), ref); err != nil {
		t.Fatalf("ref Read: %v", err)
	}

	// Rows/cols [2,6) — a small sub-region entirely inside the top-left
	// shard, intersecting only a handful of its 16 inner (2x2) chunks.
	sel := zarr.Selection{Ranges: []zarr.DimRange{{Start: 2, Stop: 6, Step: 1}, {Start: 2, Stop: 6, Step: 1}}}
	got := make([]int32, sel.NumElements())
	if err := b.Read(ctx, sel, got); err != nil {
		t.Fatalf("partial Read: %v", err)
	}
	idx := 0
	for r := int64(2); r < 6; r++ {
		for c := int64(2); c < 6; c++ {
			if got[idx] != ref[r*16+c] {
				t.Fatalf("[%d,%d] got %d want %d", r, c, got[idx], ref[r*16+c])
			}
			idx++
		}
	}

	if g := cs.gets.Load(); g != 0 {
		t.Errorf("partial read issued %d whole-chunk Get calls, want 0", g)
	}
	if suf := cs.getSuffixes.Load(); suf != 1 {
		t.Errorf("partial read issued %d GetSuffix calls, want exactly 1 (index trailer)", suf)
	}
	if r := cs.getRanges.Load(); r == 0 || r > 2 {
		t.Errorf("partial read issued %d GetRange calls, want 1-2 (coalesced inner chunks)", r)
	}
}

// rangeOnlyStore exposes only the store.Store method set of the wrapped
// store, hiding any GetSuffix it has.
type rangeOnlyStore struct{ store.Store }

// TestPartialRead_NoSuffixStore checks the partial-decode path against a
// store without GetSuffix: the sharding codec must fall back to a single
// whole-shard GetRange (then serve the index and inner chunks from it)
// rather than failing or issuing whole-chunk Gets.
func TestPartialRead_NoSuffixStore(t *testing.T) {
	ctx := context.Background()
	inner := mem.New()
	shape := []int64{16, 16}
	a, err := zarr.CreateArray[int32](ctx, inner, "arr", shape, []int64{8, 8},
		zarr.WithCodecs(codec.Spec{Name: "sharding_indexed", Configuration: json.RawMessage(shardCfg2x2)}))
	if err != nil {
		t.Fatalf("CreateArray: %v", err)
	}
	src := make([]int32, 16*16)
	for i := range src {
		src[i] = int32(i + 1)
	}
	if err := a.Write(ctx, zarr.All(shape), src); err != nil {
		t.Fatalf("Write: %v", err)
	}

	cs := &countingStore{inner: inner}
	b, err := zarr.OpenArray[int32](ctx, rangeOnlyStore{cs}, "arr")
	if err != nil {
		t.Fatalf("OpenArray: %v", err)
	}
	cs.gets.Store(0)
	sel := zarr.Selection{Ranges: []zarr.DimRange{{Start: 2, Stop: 6, Step: 1}, {Start: 2, Stop: 6, Step: 1}}}
	got := make([]int32, sel.NumElements())
	if err := b.Read(ctx, sel, got); err != nil {
		t.Fatalf("partial Read: %v", err)
	}
	idx := 0
	for r := int64(2); r < 6; r++ {
		for c := int64(2); c < 6; c++ {
			if want := src[r*16+c]; got[idx] != want {
				t.Fatalf("[%d,%d] got %d want %d", r, c, got[idx], want)
			}
			idx++
		}
	}
	if g, suf, r := cs.gets.Load(), cs.getSuffixes.Load(), cs.getRanges.Load(); g != 0 || suf != 0 || r != 1 {
		t.Errorf("partial read issued Get=%d GetSuffix=%d GetRange=%d, want 0/0/1 (one whole-shard range)", g, suf, r)
	}
}

// TestWithRangeGap_AffectsFetchCount verifies WithRangeGap actually reaches
// the sharding codec's range coalescing: a tight gap issues one GetRange per
// disjoint span, a generous gap coalesces them into one.
func TestWithRangeGap_AffectsFetchCount(t *testing.T) {
	ctx := context.Background()
	inner := mem.New()
	shape := []int64{8, 8}
	chunks := []int64{8, 8}
	a, err := zarr.CreateArray[int32](ctx, inner, "arr", shape, chunks,
		zarr.WithCodecs(codec.Spec{Name: "sharding_indexed", Configuration: json.RawMessage(shardCfg2x2)}))
	if err != nil {
		t.Fatalf("CreateArray: %v", err)
	}
	src := make([]int32, 64)
	for i := range src {
		src[i] = int32(i + 1)
	}
	if err := a.Write(ctx, zarr.All(shape), src); err != nil {
		t.Fatalf("Write: %v", err)
	}

	// Full-height, 2-of-8-column-wide selection: touches inner grid column 0
	// across all 4 grid rows of the 4x4 inner grid — 4 needed 16-byte inner
	// chunk spans, each separated by a 48-byte gap of unneeded inner chunks.
	sel := zarr.Selection{Ranges: []zarr.DimRange{
		{Start: 0, Stop: 8, Step: 1},
		{Start: 0, Stop: 2, Step: 1},
	}}

	run := func(gap int64) int64 {
		cs := &countingStore{inner: inner}
		b, err := zarr.OpenArray[int32](ctx, cs, "arr", zarr.WithRangeGap(gap))
		if err != nil {
			t.Fatalf("OpenArray: %v", err)
		}
		got := make([]int32, sel.NumElements())
		if err := b.Read(ctx, sel, got); err != nil {
			t.Fatalf("Read (gap=%d): %v", gap, err)
		}
		return cs.getRanges.Load()
	}

	noCoalesce := run(0)
	coalesce := run(1 << 20)
	if noCoalesce <= coalesce {
		t.Errorf("WithRangeGap(0) issued %d GetRange calls, want strictly more than WithRangeGap(1<<20)'s %d",
			noCoalesce, coalesce)
	}
	if coalesce != 1 {
		t.Errorf("WithRangeGap(1<<20) issued %d GetRange calls, want exactly 1 (fully coalesced)", coalesce)
	}
}

// gatedIOStore blocks every Get call until release is closed, signaling on
// entered first — but only once active is set, so metadata reads made
// during OpenArray itself (before the test arms the gate) pass through
// immediately. Used to deterministically prove WithIOConcurrency bounds the
// number of concurrently in-flight chunk fetches.
type gatedIOStore struct {
	store.Writer
	active  atomic.Bool
	entered chan struct{}
	release chan struct{}
}

func (g *gatedIOStore) Get(ctx context.Context, key string) ([]byte, error) {
	if g.active.Load() {
		g.entered <- struct{}{}
		<-g.release
	}
	return g.Writer.Get(ctx, key)
}

// TestWithIOConcurrency_BoundsInFlight proves WithIOConcurrency(N) bounds
// the number of concurrently in-flight store fetches to N, using a store
// that blocks until released so the bound can be observed deterministically
// rather than inferred from timing. The option must take effect whether it
// is passed to OpenArray or directly to CreateArray.
func TestWithIOConcurrency_BoundsInFlight(t *testing.T) {
	const ioConc = 3
	shape := []int64{32}
	chunks := []int64{2} // 16 chunks
	src := make([]int32, 32)
	for i := range src {
		src[i] = int32(i + 1)
	}
	for _, via := range []string{"OpenArray", "CreateArray"} {
		t.Run(via, func(t *testing.T) {
			ctx := context.Background()
			// entered is buffered comfortably past the chunk count so that a
			// chunk goroutine launched after the test stops draining it (i.e.
			// once the bound has been proven) can still signal without
			// blocking forever.
			gated := &gatedIOStore{Writer: mem.New(), entered: make(chan struct{}, 32), release: make(chan struct{})}
			var createOpts []zarr.CreateOption
			if via == "CreateArray" {
				createOpts = append(createOpts, zarr.WithIOConcurrency(ioConc))
			}
			b, err := zarr.CreateArray[int32](ctx, gated, "arr", shape, chunks, createOpts...)
			if err != nil {
				t.Fatalf("CreateArray: %v", err)
			}
			if err := b.Write(ctx, zarr.All(shape), src); err != nil {
				t.Fatalf("Write: %v", err)
			}
			if via == "OpenArray" {
				if b, err = zarr.OpenArray[int32](ctx, gated, "arr", zarr.WithIOConcurrency(ioConc)); err != nil {
					t.Fatalf("OpenArray: %v", err)
				}
			}
			gated.active.Store(true)

			got := make([]int32, 32)
			done := make(chan error, 1)
			go func() { done <- b.Read(ctx, zarr.All(shape), got) }()

			for i := 0; i < ioConc; i++ {
				select {
				case <-gated.entered:
				case <-time.After(2 * time.Second):
					t.Fatalf("timed out waiting for in-flight fetch %d/%d", i+1, ioConc)
				}
			}
			select {
			case <-gated.entered:
				t.Fatalf("more than %d concurrent Get calls in flight", ioConc)
			case <-time.After(100 * time.Millisecond):
			}
			close(gated.release)

			if err := <-done; err != nil {
				t.Fatalf("Read: %v", err)
			}
			for i := range src {
				if got[i] != src[i] {
					t.Fatalf("got[%d]=%d want %d", i, got[i], src[i])
				}
			}
		})
	}
}
