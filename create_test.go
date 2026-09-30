package zarr_test

import (
	"context"
	"errors"
	"testing"

	"github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/store"
	"github.com/exergy-dev/zarr-go/store/mem"
)

// TestCreateArrayAlreadyExists confirms that a second CreateArray on the same
// path fails atomically with store.ErrAlreadyExists, and that WithOverwrite
// bypasses that check.
func TestCreateArrayAlreadyExists(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	if _, err := zarr.CreateArray[float32](ctx, s, "arr",
		[]int64{4}, []int64{2}); err != nil {
		t.Fatalf("first CreateArray: %v", err)
	}
	_, err := zarr.CreateArray[float32](ctx, s, "arr", []int64{4}, []int64{2})
	if !errors.Is(err, store.ErrAlreadyExists) {
		t.Fatalf("second CreateArray: got %v, want ErrAlreadyExists", err)
	}
	// WithOverwrite still works.
	if _, err := zarr.CreateArray[float32](ctx, s, "arr",
		[]int64{4}, []int64{2}, zarr.WithOverwrite()); err != nil {
		t.Fatalf("CreateArray WithOverwrite: %v", err)
	}
}

// TestCreateGroupAlreadyExists mirrors TestCreateArrayAlreadyExists for groups.
func TestCreateGroupAlreadyExists(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	if _, err := zarr.CreateGroup(ctx, s, "g", nil); err != nil {
		t.Fatalf("first CreateGroup: %v", err)
	}
	_, err := zarr.CreateGroup(ctx, s, "g", nil)
	if !errors.Is(err, store.ErrAlreadyExists) {
		t.Fatalf("second CreateGroup: got %v, want ErrAlreadyExists", err)
	}
}

// TestCreateArrayDefaultFillValueByDtype guards against CreateArray[bool] and
// CreateArray[complexN] always failing when no WithFillValue is given: the
// blanket JSON literal "0" (valid for numeric dtypes) is invalid JSON for
// bool (needs true/false) and complex (needs a 2-element array).
func TestCreateArrayDefaultFillValueByDtype(t *testing.T) {
	ctx := context.Background()

	t.Run("bool", func(t *testing.T) {
		s := mem.New()
		if _, err := zarr.CreateArray[bool](ctx, s, "b", []int64{4}, []int64{2}); err != nil {
			t.Fatalf("CreateArray[bool]: %v", err)
		}
	})
	t.Run("complex64", func(t *testing.T) {
		s := mem.New()
		if _, err := zarr.CreateArray[complex64](ctx, s, "c", []int64{4}, []int64{2}); err != nil {
			t.Fatalf("CreateArray[complex64]: %v", err)
		}
	})
	t.Run("complex128", func(t *testing.T) {
		s := mem.New()
		if _, err := zarr.CreateArray[complex128](ctx, s, "c", []int64{4}, []int64{2}); err != nil {
			t.Fatalf("CreateArray[complex128]: %v", err)
		}
	})
}

// TestWithOverwriteDeletesStaleChunks guards against WithOverwrite leaving
// old chunk bytes behind when an array is re-created with a different
// dtype/shape at the same path: without cleanup, reads of the "new" array
// would decode the stale bytes as garbage instead of fill.
func TestWithOverwriteDeletesStaleChunks(t *testing.T) {
	ctx := context.Background()
	s := mem.New()

	a, err := zarr.CreateArray[float32](ctx, s, "arr", []int64{4}, []int64{2})
	if err != nil {
		t.Fatalf("first CreateArray: %v", err)
	}
	if err := a.Write(ctx, zarr.All([]int64{4}), []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	// Sanity: chunk keys exist in the store before the overwrite.
	if ok, err := s.Exists(ctx, "arr/c/0"); err != nil || !ok {
		t.Fatalf("precondition: chunk arr/c/0 should exist, ok=%v err=%v", ok, err)
	}

	b, err := zarr.CreateArray[int32](ctx, s, "arr", []int64{8}, []int64{4}, zarr.WithOverwrite())
	if err != nil {
		t.Fatalf("overwrite CreateArray: %v", err)
	}
	if ok, err := s.Exists(ctx, "arr/c/0"); err != nil || ok {
		t.Fatalf("stale chunk arr/c/0 should have been deleted, ok=%v err=%v", ok, err)
	}
	dst := make([]int32, 8)
	if err := b.Read(ctx, zarr.All([]int64{8}), dst); err != nil {
		t.Fatalf("Read: %v", err)
	}
	for i, v := range dst {
		if v != 0 {
			t.Fatalf("dst[%d] = %d, want 0 (fill) — stale chunk bytes leaked through", i, v)
		}
	}
}

// TestWithV2ChunkKeys checks the new option wires the dead
// keyEncName/keySep createOptions fields: without it chunks were always
// written at the v3 default layout regardless of caller intent.
func TestWithV2ChunkKeys(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	a, err := zarr.CreateArray[int32](ctx, s, "arr", []int64{4}, []int64{2}, zarr.WithV2ChunkKeys())
	if err != nil {
		t.Fatalf("CreateArray: %v", err)
	}
	if err := a.Write(ctx, zarr.All([]int64{4}), []int32{1, 2, 3, 4}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if ok, err := s.Exists(ctx, "arr/0"); err != nil || !ok {
		t.Fatalf("expected v2-style chunk key arr/0 to exist, ok=%v err=%v", ok, err)
	}
	if ok, _ := s.Exists(ctx, "arr/c/0"); ok {
		t.Fatal("did not expect a v3-default-style chunk key arr/c/0")
	}

	// Re-opening must round-trip the v2 chunk_key_encoding and read the data back.
	b, err := zarr.OpenArray[int32](ctx, s, "arr")
	if err != nil {
		t.Fatalf("OpenArray: %v", err)
	}
	dst := make([]int32, 4)
	if err := b.Read(ctx, zarr.All([]int64{4}), dst); err != nil {
		t.Fatalf("Read: %v", err)
	}
	want := []int32{1, 2, 3, 4}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d] = %d, want %d", i, dst[i], want[i])
		}
	}
}

// TestWithChunkKeySeparatorOverridesV2Default checks that an explicit
// WithChunkKeySeparator after WithV2ChunkKeys wins over the "." default.
func TestWithChunkKeySeparatorOverridesV2Default(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	a, err := zarr.CreateArray[int32](ctx, s, "arr", []int64{4, 4}, []int64{2, 2},
		zarr.WithV2ChunkKeys(), zarr.WithChunkKeySeparator("/"))
	if err != nil {
		t.Fatalf("CreateArray: %v", err)
	}
	src := make([]int32, 16)
	for i := range src {
		src[i] = int32(i)
	}
	if err := a.Write(ctx, zarr.All([]int64{4, 4}), src); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if ok, err := s.Exists(ctx, "arr/0/0"); err != nil || !ok {
		t.Fatalf("expected chunk key arr/0/0 (slash separator), ok=%v err=%v", ok, err)
	}
}

// TestCreateArrayReadOnlyStore checks CreateArray and CreateGroup both
// return the exported, errors.Is-matchable ErrReadOnly sentinel against a
// store that doesn't implement store.WritableStore.
func TestCreateArrayReadOnlyStore(t *testing.T) {
	ctx := context.Background()
	ro := readOnlyStore{mem.New()}
	if _, err := zarr.CreateArray[int32](ctx, ro, "arr", []int64{4}, []int64{2}); !errors.Is(err, zarr.ErrReadOnly) {
		t.Fatalf("CreateArray: got %v, want ErrReadOnly", err)
	}
	if _, err := zarr.CreateGroup(ctx, ro, "g", nil); !errors.Is(err, zarr.ErrReadOnly) {
		t.Fatalf("CreateGroup: got %v, want ErrReadOnly", err)
	}
}

// TestArrayWriteReadOnlyStore checks Array.Write returns the exported
// ErrReadOnly sentinel (not an unexported/ad-hoc error) against a
// read-only store.
func TestArrayWriteReadOnlyStore(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	if _, err := zarr.CreateArray[int32](ctx, s, "arr", []int64{4}, []int64{2}); err != nil {
		t.Fatalf("CreateArray: %v", err)
	}
	ro, err := zarr.OpenArray[int32](ctx, readOnlyStore{s}, "arr")
	if err != nil {
		t.Fatalf("OpenArray via read-only store: %v", err)
	}
	err = ro.Write(ctx, zarr.All([]int64{4}), []int32{1, 2, 3, 4})
	if !errors.Is(err, zarr.ErrReadOnly) {
		t.Fatalf("Write: got %v, want ErrReadOnly", err)
	}
}

// readOnlyStore wraps a store.Store but deliberately does NOT implement
// store.WritableStore, so type-asserting for it fails as intended.
type readOnlyStore struct {
	store.Store
}

// TestGroupAttrsDoesNotAliasCaller checks that WithAttributes/CreateGroup's
// attrs param and Group.Attrs()'s return value don't let a caller mutate the
// group's in-memory metadata (which would silently diverge from the
// already-persisted zarr.json) via the map they passed in or got back.
func TestGroupAttrsDoesNotAliasCaller(t *testing.T) {
	ctx := context.Background()
	s := mem.New()

	in := map[string]any{"k": "v1"}
	g, err := zarr.CreateGroup(ctx, s, "g", in)
	if err != nil {
		t.Fatalf("CreateGroup: %v", err)
	}
	in["k"] = "mutated-after-create"
	if got := g.Attrs()["k"]; got != "v1" {
		t.Fatalf("Attrs()[k] = %v, want v1 (CreateGroup aliased the caller's map)", got)
	}

	out := g.Attrs()
	out["k"] = "mutated-via-returned-map"
	if got := g.Attrs()["k"]; got != "v1" {
		t.Fatalf("Attrs()[k] = %v, want v1 (Attrs() returned an alias, not a copy)", got)
	}
}

// TestCreateArrayRank0 checks a rank-0 (scalar) array round-trips through
// create, write, and read.
func TestCreateArrayRank0(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	a, err := zarr.CreateArray[float64](ctx, s, "scalar", nil, nil)
	if err != nil {
		t.Fatalf("CreateArray rank-0: %v", err)
	}
	sel := zarr.All(nil)
	if n := sel.NumElements(); n != 1 {
		t.Fatalf("NumElements() = %d, want 1", n)
	}
	if err := a.Write(ctx, sel, []float64{3.5}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if ok, err := s.Exists(ctx, "scalar/c"); err != nil || !ok {
		t.Fatalf("expected chunk key scalar/c to exist, ok=%v err=%v", ok, err)
	}

	b, err := zarr.OpenArray[float64](ctx, s, "scalar")
	if err != nil {
		t.Fatalf("OpenArray rank-0: %v", err)
	}
	dst := make([]float64, 1)
	if err := b.Read(ctx, sel, dst); err != nil {
		t.Fatalf("Read: %v", err)
	}
	if dst[0] != 3.5 {
		t.Fatalf("dst[0] = %v, want 3.5", dst[0])
	}
}
