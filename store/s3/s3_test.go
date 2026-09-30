package s3

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/aws/smithy-go"
	smithyhttp "github.com/aws/smithy-go/transport/http"

	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	"github.com/exergy-dev/zarr-go/internal/cloudstore"
	"github.com/exergy-dev/zarr-go/internal/cloudstore/cloudstoretest"
	"github.com/exergy-dev/zarr-go/store"
	"github.com/exergy-dev/zarr-go/store/storetest"
)

func TestIsNotFound(t *testing.T) {
	if isNotFound(nil) {
		t.Error("isNotFound(nil) should be false")
	}
	if !isNotFound(&types.NoSuchKey{}) {
		t.Error("isNotFound(NoSuchKey) should be true")
	}
	if !isNotFound(&types.NotFound{}) {
		t.Error("isNotFound(NotFound) should be true")
	}
	if isNotFound(errors.New("random")) {
		t.Error("isNotFound(random) should be false")
	}

	// Wrap NoSuchKey to verify errors.As traversal.
	wrapped := fmt.Errorf("ctx: %w", &types.NoSuchKey{})
	if !isNotFound(wrapped) {
		t.Error("isNotFound(wrapped NoSuchKey) should be true")
	}

	// 404 response error.
	re404 := &smithyhttp.ResponseError{
		Response: &smithyhttp.Response{Response: &http.Response{StatusCode: 404}},
		Err:      &smithy.GenericAPIError{Code: "NotFound"},
	}
	if !isNotFound(re404) {
		t.Error("isNotFound(404 ResponseError) should be true")
	}

	re500 := &smithyhttp.ResponseError{
		Response: &smithyhttp.Response{Response: &http.Response{StatusCode: 500}},
		Err:      &smithy.GenericAPIError{Code: "Internal"},
	}
	if isNotFound(re500) {
		t.Error("isNotFound(500 ResponseError) should be false")
	}
}

func TestNotFoundErrIsErrKeyNotFound(t *testing.T) {
	err := notFoundErr("k", &types.NoSuchKey{})
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

// fakeStore returns a Store pointed at a credential-free httptest endpoint,
// isolated from the host's AWS config and credentials.
func fakeStore(t *testing.T, endpoint string) *Store {
	t.Helper()
	t.Setenv("AWS_ACCESS_KEY_ID", "test")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "test")
	t.Setenv("AWS_SESSION_TOKEN", "")
	t.Setenv("AWS_PROFILE", "")
	t.Setenv("AWS_CONFIG_FILE", filepath.Join(t.TempDir(), "config"))
	t.Setenv("AWS_SHARED_CREDENTIALS_FILE", filepath.Join(t.TempDir(), "credentials"))
	t.Setenv("AWS_EC2_METADATA_DISABLED", "true")
	s, err := New(context.Background(), Config{
		Bucket:    "b",
		Region:    "us-east-1",
		Endpoint:  endpoint,
		PathStyle: true,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

// TestGetRange_HugeLengthRangeHeader: off+n past the end must clamp to the
// object size, including when off+n-1 overflows int64 (n = MaxInt64).
func TestGetRange_HugeLengthRangeHeader(t *testing.T) {
	url, ranges := cloudstoretest.RangeServer(t, "Range", []byte("0123456789"))
	s := fakeStore(t, url)
	got, err := s.GetRange(context.Background(), "k", 5, math.MaxInt64)
	if err != nil {
		t.Fatalf("GetRange(5, MaxInt64): %v (Range headers sent: %q)", err, ranges())
	}
	if string(got) != "56789" {
		t.Fatalf("GetRange(5, MaxInt64) = %q, want %q", got, "56789")
	}
	if r, want := ranges(), "bytes=5-9223372036854775806"; len(r) != 1 || r[0] != want {
		t.Fatalf("range headers sent = %q, want [%q]", r, want)
	}
}

// TestEmptyKeyRejected: single-key operations must reject key "" before
// issuing any request, rather than addressing the object named exactly
// Prefix (outside the store's Prefix+"/" namespace).
func TestEmptyKeyRejected(t *testing.T) {
	url, requests := cloudstoretest.RangeServer(t, "Range", []byte("x"))
	s := fakeStore(t, url)
	s.prefix = "data"
	ctx := context.Background()
	ops := map[string]func() error{
		"Get":            func() error { _, err := s.Get(ctx, ""); return err },
		"GetRange":       func() error { _, err := s.GetRange(ctx, "", 0, 1); return err },
		"GetSuffix":      func() error { _, err := s.GetSuffix(ctx, "/", 1); return err },
		"Set":            func() error { return s.Set(ctx, "", []byte("x")) },
		"SetIfNotExists": func() error { _, err := s.SetIfNotExists(ctx, "", []byte("x")); return err },
		"Delete":         func() error { return s.Delete(ctx, "") },
		"Exists":         func() error { _, err := s.Exists(ctx, ""); return err },
	}
	for name, op := range ops {
		if err := op(); !errors.Is(err, cloudstore.ErrEmptyKey) {
			t.Errorf("%s(\"\") = %v, want ErrEmptyKey", name, err)
		}
	}
	if r := requests(); len(r) != 0 {
		t.Errorf("empty-key operations issued %d requests, want 0", len(r))
	}
}

// integration tests --------------------------------------------------------

const envEndpoint = "ZARRGO_S3_TEST_ENDPOINT"

// integrationStore returns a Store wired against a test endpoint, or skips.
func integrationStore(t *testing.T) *Store {
	t.Helper()
	endpoint := os.Getenv(envEndpoint)
	if endpoint == "" {
		t.Skipf("set %s to run S3 integration tests", envEndpoint)
	}
	if _, err := url.Parse(endpoint); err != nil {
		t.Fatalf("invalid %s: %v", envEndpoint, err)
	}
	bucket := os.Getenv("ZARRGO_S3_TEST_BUCKET")
	if bucket == "" {
		bucket = "zarrgo-test"
	}
	region := os.Getenv("ZARRGO_S3_TEST_REGION")
	if region == "" {
		region = "us-east-1"
	}
	s, err := New(context.Background(), Config{
		Bucket:    bucket,
		Region:    region,
		Endpoint:  endpoint,
		PathStyle: true,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
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

func TestIntegration_Conformance(t *testing.T) {
	storetest.RunConformance(t, func(t *testing.T) store.Store {
		return integrationStore(t)
	}, storetest.Options{})
}
