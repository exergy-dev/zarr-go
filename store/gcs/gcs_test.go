package gcs

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"testing"

	"cloud.google.com/go/storage"
	"google.golang.org/api/googleapi"

	"github.com/exergy-dev/zarr-go/internal/cloudstore/cloudstoretest"
	"github.com/exergy-dev/zarr-go/store"
	"github.com/exergy-dev/zarr-go/store/storetest"
)

func TestIsPreconditionFailed(t *testing.T) {
	if isPreconditionFailed(nil) {
		t.Error("isPreconditionFailed(nil) should be false")
	}
	if !isPreconditionFailed(&googleapi.Error{Code: 412}) {
		t.Error("isPreconditionFailed(412) should be true")
	}
	if isPreconditionFailed(&googleapi.Error{Code: 404}) {
		t.Error("isPreconditionFailed(404) should be false")
	}
	if isPreconditionFailed(errors.New("random")) {
		t.Error("isPreconditionFailed(random) should be false")
	}
	// Wrapped 412 should still be detected via errors.As.
	wrapped := fmt.Errorf("ctx: %w", &googleapi.Error{Code: 412})
	if !isPreconditionFailed(wrapped) {
		t.Error("isPreconditionFailed(wrapped 412) should be true")
	}
}

func TestNotFoundErrIsErrKeyNotFound(t *testing.T) {
	err := notFoundErr("k")
	if !errors.Is(err, store.ErrKeyNotFound) {
		t.Errorf("expected errors.Is(err, store.ErrKeyNotFound), got %v", err)
	}
	// Verify it also works through os.ErrNotExist via the sentinel's Is.
	if !errors.Is(err, os.ErrNotExist) {
		t.Errorf("expected errors.Is(err, os.ErrNotExist), got %v", err)
	}
}

func TestNew_RequiresBucket(t *testing.T) {
	_, err := New(context.Background(), Config{})
	if err == nil {
		t.Fatal("expected error when Bucket is empty")
	}
}

// fakeStore returns a Store for bucket "b" on a cloudstoretest.Server.
func fakeStore(t *testing.T, url string) *Store {
	t.Helper()
	t.Setenv("STORAGE_EMULATOR_HOST", url)
	client, err := storage.NewClient(context.Background())
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return &Store{client: client, bucket: client.Bucket("b"), name: "b"}
}

// TestFakeConformance runs the read side of the store contract (Basics,
// GetRange, GetSuffix) against the credential-free fake.
func TestFakeConformance(t *testing.T) {
	srv := cloudstoretest.NewServer(t, "Range")
	storetest.RunConformance(t, func(t *testing.T) store.Store {
		return fakeStore(t, srv.URL)
	}, storetest.Options{
		ReadOnly: true,
		NoList:   true,
		Seed:     func(_ *testing.T, _ store.Store, key string, value []byte) { srv.Put(key, value) },
	})
}

// TestGetRange_HugeLengthRangeHeader: off+n past the end must clamp to the
// object size, including when off+n-1 overflows int64 (n = MaxInt64). The
// GCS HTTP reader builds "bytes=off-(off+length-1)" itself, so an unclamped
// length overflows inside the SDK.
func TestGetRange_HugeLengthRangeHeader(t *testing.T) {
	srv := cloudstoretest.NewServer(t, "Range")
	srv.Put("k", []byte("0123456789"))
	s := fakeStore(t, srv.URL)
	got, err := s.GetRange(context.Background(), "k", 5, math.MaxInt64)
	if err != nil {
		t.Fatalf("GetRange(5, MaxInt64): %v (Range headers sent: %q)", err, srv.Ranges())
	}
	if string(got) != "56789" {
		t.Fatalf("GetRange(5, MaxInt64) = %q, want %q", got, "56789")
	}
	if r, want := srv.Ranges(), "bytes=5-9223372036854775806"; len(r) != 1 || r[0] != want {
		t.Fatalf("range headers sent = %q, want [%q]", r, want)
	}
}

// integration tests --------------------------------------------------------

const envBucket = "ZARRGO_GCS_TEST_BUCKET"

// integrationStore returns a Store wired against a real GCS bucket, or skips.
func integrationStore(t *testing.T) *Store {
	t.Helper()
	bucket := os.Getenv(envBucket)
	if bucket == "" {
		t.Skipf("set %s to run GCS integration tests", envBucket)
	}
	prefix := os.Getenv("ZARRGO_GCS_TEST_PREFIX")
	s, err := New(context.Background(), Config{
		Bucket: bucket,
		Prefix: prefix,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestIntegration_RoundTrip(t *testing.T) {
	s := integrationStore(t)
	ctx := context.Background()

	key := "round/trip/key"
	value := []byte("hello, zarr")

	if err := s.Set(ctx, key, value); err != nil {
		t.Fatalf("Set: %v", err)
	}
	t.Cleanup(func() { _ = s.Delete(ctx, key) })

	ok, err := s.Exists(ctx, key)
	if err != nil || !ok {
		t.Fatalf("Exists: ok=%v err=%v", ok, err)
	}

	got, err := s.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if string(got) != string(value) {
		t.Fatalf("Get got %q, want %q", got, value)
	}

	// Range read.
	rng, err := s.GetRange(ctx, key, 7, 4)
	if err != nil {
		t.Fatalf("GetRange: %v", err)
	}
	if string(rng) != "zarr" {
		t.Fatalf("GetRange got %q, want %q", rng, "zarr")
	}

	if err := s.Delete(ctx, key); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	ok, err = s.Exists(ctx, key)
	if err != nil || ok {
		t.Fatalf("Exists after delete: ok=%v err=%v", ok, err)
	}

	if _, err := s.Get(ctx, key); !errors.Is(err, store.ErrKeyNotFound) {
		t.Fatalf("Get missing: want ErrKeyNotFound, got %v", err)
	}
}

func TestIntegration_List(t *testing.T) {
	s := integrationStore(t)
	ctx := context.Background()

	keys := []string{"list/a", "list/b", "list/c/d"}
	for _, k := range keys {
		if err := s.Set(ctx, k, []byte("x")); err != nil {
			t.Fatalf("Set %q: %v", k, err)
		}
		k := k
		t.Cleanup(func() { _ = s.Delete(ctx, k) })
	}

	var got []string
	for k, err := range s.List(ctx, "list/") {
		if err != nil {
			t.Fatalf("List: %v", err)
		}
		got = append(got, k)
	}
	slices.Sort(got)
	want := []string{"list/a", "list/b", "list/c/d"}
	if len(got) != len(want) {
		t.Fatalf("List got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("List[%d]=%q, want %q", i, got[i], want[i])
		}
	}
}

func TestIntegration_DeleteMissing(t *testing.T) {
	s := integrationStore(t)
	ctx := context.Background()
	if err := s.Delete(ctx, "definitely/does/not/exist"); err != nil {
		t.Fatalf("Delete missing should be nil, got %v", err)
	}
}

func TestIntegration_SetIfNotExists(t *testing.T) {
	s := integrationStore(t)
	ctx := context.Background()

	key := "sine/key"
	t.Cleanup(func() { _ = s.Delete(ctx, key) })
	_ = s.Delete(ctx, key)

	ok, err := s.SetIfNotExists(ctx, key, []byte("first"))
	if err != nil {
		t.Fatalf("SetIfNotExists first: %v", err)
	}
	if !ok {
		t.Fatal("SetIfNotExists first: expected true")
	}

	ok, err = s.SetIfNotExists(ctx, key, []byte("second"))
	if err != nil {
		t.Fatalf("SetIfNotExists second: %v", err)
	}
	if ok {
		t.Fatal("SetIfNotExists second: expected false (already exists)")
	}

	got, err := s.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if string(got) != "first" {
		t.Fatalf("Get got %q, want %q", got, "first")
	}
}

// silence unused import when integration tests are skipped.
var _ = storage.ErrObjectNotExist

func TestIntegration_Conformance(t *testing.T) {
	storetest.RunConformance(t, func(t *testing.T) store.Store {
		return integrationStore(t)
	}, storetest.Options{})
}
