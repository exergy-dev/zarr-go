package azblob

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"testing"

	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/bloberror"

	"github.com/rkm/zarr-go/store"
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
	if isNotFound(errors.New("random")) {
		t.Error("isNotFound(random) should be false")
	}
}

func TestNotFoundErrIsErrKeyNotFound(t *testing.T) {
	err := notFoundErr("k", fmt.Errorf("orig: %w", errors.New("BlobNotFound")))
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

func TestNew_RequiresAccountNameOrServiceURL(t *testing.T) {
	_, err := New(context.Background(), Config{Container: "c", AccountKey: "k"})
	if err == nil {
		t.Fatal("expected error when AccountName and ServiceURL are both empty")
	}
}

func TestNew_RequiresAccountKey(t *testing.T) {
	_, err := New(context.Background(), Config{AccountName: "x", Container: "c"})
	if err == nil {
		t.Fatal("expected error when AccountKey is empty (azidentity not supported)")
	}
}

func TestReadSeekCloser(t *testing.T) {
	r := newReadSeekCloser([]byte("hello, zarr"))
	buf := make([]byte, 5)
	n, err := r.Read(buf)
	if err != nil || n != 5 || string(buf[:n]) != "hello" {
		t.Fatalf("Read: n=%d buf=%q err=%v", n, buf[:n], err)
	}
	if pos, err := r.Seek(7, io.SeekStart); err != nil || pos != 7 {
		t.Fatalf("Seek(7,Start): pos=%d err=%v", pos, err)
	}
	rest, err := io.ReadAll(r)
	if err != nil || string(rest) != "zarr" {
		t.Fatalf("ReadAll after seek: %q err=%v", rest, err)
	}
	if err := r.Close(); err != nil {
		t.Fatalf("Close: %v", err)
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
