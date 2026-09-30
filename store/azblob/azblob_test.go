package azblob

import (
	"context"
	"encoding/base64"
	"errors"
	"math"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/bloberror"

	"github.com/exergy-dev/zarr-go/internal/cloudstore/cloudstoretest"
	"github.com/exergy-dev/zarr-go/store"
	"github.com/exergy-dev/zarr-go/store/storetest"
)

func TestIsNotFound(t *testing.T) {
	if isNotFound(nil) {
		t.Error("isNotFound(nil) should be false")
	}
	if isNotFound(errors.New("random")) {
		t.Error("isNotFound(random) should be false")
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

func TestBloberrorCodes(t *testing.T) {
	// Sanity check that the constants we depend on exist with the values
	// we expect.
	if string(bloberror.BlobNotFound) != "BlobNotFound" {
		t.Errorf("bloberror.BlobNotFound = %q, want %q",
			string(bloberror.BlobNotFound), "BlobNotFound")
	}
}

func TestNew_RequiresContainer(t *testing.T) {
	_, err := New(context.Background(), Config{AccountName: "x", AccountKey: "key"})
	if err == nil {
		t.Fatal("expected error when Container is empty")
	}
}

// TestNew_RequiresAccountName: shared-key auth signs every request with the
// account name, so it is required even when ServiceURL gives the endpoint.
func TestNew_RequiresAccountName(t *testing.T) {
	key := base64.StdEncoding.EncodeToString([]byte("key"))
	for _, cfg := range []Config{
		{Container: "c", AccountKey: key},
		{Container: "c", AccountKey: key, ServiceURL: "https://acct.blob.core.windows.net/"},
	} {
		_, err := New(context.Background(), cfg)
		if err == nil || !strings.Contains(err.Error(), "AccountName is required") {
			t.Errorf("New(ServiceURL=%q, no AccountName) error = %v, want \"AccountName is required\"", cfg.ServiceURL, err)
		}
	}
}

func TestNew_RequiresAccountKey(t *testing.T) {
	_, err := New(context.Background(), Config{AccountName: "x", Container: "c"})
	if err == nil {
		t.Fatal("expected error when AccountKey is empty (azidentity not supported)")
	}
}

// TestGetRange_HugeLengthRangeHeader: off+n past the end must clamp to the
// blob size, including when off+n-1 overflows int64 (n = MaxInt64). The SDK
// formats x-ms-range as "bytes=Offset-(Offset+Count-1)" itself, so an
// unclamped Count overflows inside the SDK.
func TestGetRange_HugeLengthRangeHeader(t *testing.T) {
	url, ranges := cloudstoretest.RangeServer(t, "x-ms-range", []byte("0123456789"))
	s, err := New(context.Background(), Config{
		AccountName: "acct",
		AccountKey:  base64.StdEncoding.EncodeToString([]byte("key")),
		Container:   "c",
		ServiceURL:  url,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got, err := s.GetRange(context.Background(), "k", 5, math.MaxInt64)
	if err != nil {
		t.Fatalf("GetRange(5, MaxInt64): %v (x-ms-range headers sent: %q)", err, ranges())
	}
	if string(got) != "56789" {
		t.Fatalf("GetRange(5, MaxInt64) = %q, want %q", got, "56789")
	}
	if r, want := ranges(), "bytes=5-9223372036854775806"; len(r) != 1 || r[0] != want {
		t.Fatalf("range headers sent = %q, want [%q]", r, want)
	}
}

// integration tests --------------------------------------------------------

const (
	envAccount   = "ZARRGO_AZBLOB_TEST_ACCOUNT"
	envKey       = "ZARRGO_AZBLOB_TEST_KEY"
	envContainer = "ZARRGO_AZBLOB_TEST_CONTAINER"
	envURL       = "ZARRGO_AZBLOB_TEST_URL"
)

// integrationStore returns a Store wired against a test account, or skips.
func integrationStore(t *testing.T) *Store {
	t.Helper()
	account := os.Getenv(envAccount)
	if account == "" {
		t.Skipf("set %s to run azblob integration tests", envAccount)
	}
	container := os.Getenv(envContainer)
	if container == "" {
		container = "zarrgo-test"
	}
	s, err := New(context.Background(), Config{
		AccountName: account,
		AccountKey:  os.Getenv(envKey),
		Container:   container,
		ServiceURL:  os.Getenv(envURL),
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

func TestIntegration_SetIfNotExists(t *testing.T) {
	s := integrationStore(t)
	ctx := context.Background()

	key := "sine/key"
	t.Cleanup(func() { _ = s.Delete(ctx, key) })
	_ = s.Delete(ctx, key)

	ok, err := s.SetIfNotExists(ctx, key, []byte("first"))
	if err != nil || !ok {
		t.Fatalf("SetIfNotExists first: ok=%v err=%v", ok, err)
	}
	ok, err = s.SetIfNotExists(ctx, key, []byte("second"))
	if err != nil || ok {
		t.Fatalf("SetIfNotExists second: ok=%v err=%v (want false, nil)", ok, err)
	}
	got, err := s.Get(ctx, key)
	if err != nil || string(got) != "first" {
		t.Fatalf("Get after SetIfNotExists: got=%q err=%v", got, err)
	}
}

func TestIntegration_Conformance(t *testing.T) {
	storetest.RunConformance(t, func(t *testing.T) store.Store {
		return integrationStore(t)
	}, storetest.Options{})
}
