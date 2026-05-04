package s3

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"slices"
	"testing"

	"github.com/aws/smithy-go"
	smithyhttp "github.com/aws/smithy-go/transport/http"

	"github.com/rkm/zarr-go/store"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
)

func TestJoinKey(t *testing.T) {
	cases := []struct {
		prefix, key, want string
	}{
		{"", "foo", "foo"},
		{"", "/foo", "foo"},
		{"p", "foo", "p/foo"},
		{"p/", "foo", "p/foo"},
		{"/p/", "foo", "p/foo"},
		{"/p/", "/foo", "p/foo"},
		{"a/b", "c/d", "a/b/c/d"},
		{"p", "", "p"},
		{"", "", ""},
		{"p", "/", "p"},
	}
	for _, c := range cases {
		got := joinKey(c.prefix, c.key)
		if got != c.want {
			t.Errorf("joinKey(%q, %q) = %q, want %q", c.prefix, c.key, got, c.want)
		}
	}
}

func TestStripPrefix(t *testing.T) {
	cases := []struct {
		prefix, key, want string
	}{
		{"", "foo", "foo"},
		{"p", "p/foo", "foo"},
		{"p", "p/foo/bar", "foo/bar"},
		{"a/b", "a/b/c", "c"},
		// key without prefix is returned as-is.
		{"p", "other/foo", "other/foo"},
	}
	for _, c := range cases {
		got := stripPrefix(c.prefix, c.key)
		if got != c.want {
			t.Errorf("stripPrefix(%q, %q) = %q, want %q", c.prefix, c.key, got, c.want)
		}
	}
}

func TestTrimSlashes(t *testing.T) {
	cases := map[string]string{
		"":         "",
		"/":        "",
		"//a//":    "a",
		"a/b":      "a/b",
		"/a/b/":    "a/b",
		"///a/b//": "a/b",
	}
	for in, want := range cases {
		if got := trimSlashes(in); got != want {
			t.Errorf("trimSlashes(%q) = %q, want %q", in, got, want)
		}
	}
}

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
