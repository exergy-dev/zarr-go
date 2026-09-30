package zarr_test

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/store"
	"github.com/exergy-dev/zarr-go/store/cache"
	"github.com/exergy-dev/zarr-go/store/fs"
	httpstore "github.com/exergy-dev/zarr-go/store/http"
)

// readOnlyStore exposes only the read side of a store.Writer.
type readOnlyStore struct {
	store.Store
}

// TestReadOnlyStoresReturnErrReadOnly: every write entry point must fail
// with an errors.Is-matchable zarr.ErrReadOnly on a read-only store — the
// http store, a cache wrapping it, a cache.New over a writable store, and
// a plain read-only wrapper — rather than a backend-specific error or a
// wrapped failure from the first write attempt.
func TestReadOnlyStoresReturnErrReadOnly(t *testing.T) {
	ctx := context.Background()

	// Author an array on disk, then serve that directory over HTTP.
	dir := t.TempDir()
	disk, err := fs.New(dir)
	if err != nil {
		t.Fatalf("fs.New: %v", err)
	}
	if _, err := zarr.CreateArray[int32](ctx, disk, "arr", []int64{4}, []int64{2}); err != nil {
		t.Fatalf("CreateArray on fs: %v", err)
	}
	srv := httptest.NewServer(http.FileServer(http.Dir(dir)))
	t.Cleanup(srv.Close)
	hs, err := httpstore.New(httpstore.Config{BaseURL: srv.URL + "/"})
	if err != nil {
		t.Fatalf("http.New: %v", err)
	}

	for name, s := range map[string]store.Store{
		"http":       hs,
		"cache/http": cache.New(hs, cache.Config{}),
		"cache/fs":   cache.New(disk, cache.Config{}),
		"wrapper/fs": readOnlyStore{disk},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := zarr.CreateArray[int32](ctx, s, "new", []int64{4}, []int64{2}); !errors.Is(err, zarr.ErrReadOnly) {
				t.Errorf("CreateArray: got %v, want ErrReadOnly", err)
			}
			if _, err := zarr.CreateArray[int32](ctx, s, "arr", []int64{4}, []int64{2}, zarr.WithOverwrite()); !errors.Is(err, zarr.ErrReadOnly) {
				t.Errorf("CreateArray(WithOverwrite): got %v, want ErrReadOnly", err)
			}
			if _, err := zarr.CreateGroup(ctx, s, "g"); !errors.Is(err, zarr.ErrReadOnly) {
				t.Errorf("CreateGroup: got %v, want ErrReadOnly", err)
			}
			a, err := zarr.OpenArray[int32](ctx, s, "arr")
			if err != nil {
				t.Fatalf("OpenArray: %v", err)
			}
			if err := a.Write(ctx, zarr.All([]int64{4}), []int32{1, 2, 3, 4}); !errors.Is(err, zarr.ErrReadOnly) {
				t.Errorf("Write: got %v, want ErrReadOnly", err)
			}
		})
	}
}
