package zarr_test

import (
	"context"
	"errors"
	"testing"

	"github.com/rkm/zarr-go"
	"github.com/rkm/zarr-go/store"
	"github.com/rkm/zarr-go/store/mem"
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
