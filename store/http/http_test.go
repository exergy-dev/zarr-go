package http

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/exergy-dev/zarr-go/store"
	"github.com/exergy-dev/zarr-go/store/storetest"
)

// fileServerStore pairs a Store with the on-disk directory its backing
// httptest server serves, so the conformance suite's Seed callback can write
// files directly instead of going through Set (which this read-only store
// doesn't support). Embedding *Store promotes Get/GetRange/GetSuffix/etc.
type fileServerStore struct {
	*Store
	dir string
}

// newFileServerStore serves dir (RFC 7233 Range support, including suffix
// ranges and 416, all via the standard library) as the backing store.
func newFileServerStore(t *testing.T) store.Store {
	t.Helper()
	dir := t.TempDir()
	srv := httptest.NewServer(http.FileServer(http.Dir(dir)))
	t.Cleanup(srv.Close)
	s, err := New(Config{BaseURL: srv.URL + "/"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return fileServerStore{Store: s, dir: dir}
}

func TestIntegration_Conformance(t *testing.T) {
	storetest.RunConformance(t, newFileServerStore, storetest.Options{
		ReadOnly: true,
		NoList:   true, // the http store has no listing protocol (ErrListUnsupported)
		Seed: func(t *testing.T, s store.Store, key string, value []byte) {
			t.Helper()
			fs := s.(fileServerStore)
			path := filepath.Join(fs.dir, filepath.FromSlash(key))
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				t.Fatalf("seed MkdirAll: %v", err)
			}
			if err := os.WriteFile(path, value, 0o644); err != nil {
				t.Fatalf("seed WriteFile: %v", err)
			}
		},
	})
}

// newTestStore spins up an httptest.Server backed by data and returns a Store
// pointed at it, plus a cleanup func.
func newTestStore(t *testing.T, data map[string][]byte, headers http.Header, requireHeader [2]string) (*Store, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if requireHeader[0] != "" {
			if got := r.Header.Get(requireHeader[0]); got != requireHeader[1] {
				http.Error(w, fmt.Sprintf("missing/bad header %s: got %q want %q", requireHeader[0], got, requireHeader[1]), http.StatusForbidden)
				return
			}
		}
		// Strip any leading slash so we can match raw map keys.
		key := strings.TrimPrefix(r.URL.Path, "/")
		body, ok := data[key]
		if !ok {
			http.NotFound(w, r)
			return
		}
		switch r.Method {
		case http.MethodHead:
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			w.WriteHeader(http.StatusOK)
		case http.MethodGet:
			if rng := r.Header.Get("Range"); rng != "" {
				// Parse "bytes=A-B".
				var a, b int64
				if _, err := fmt.Sscanf(rng, "bytes=%d-%d", &a, &b); err == nil {
					if a < 0 || a >= int64(len(body)) {
						w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
						return
					}
					end := b + 1
					if end > int64(len(body)) {
						end = int64(len(body))
					}
					w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", a, end-1, len(body)))
					w.WriteHeader(http.StatusPartialContent)
					_, _ = w.Write(body[a:end])
					return
				}
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(body)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}))
	t.Cleanup(srv.Close)

	cfg := Config{
		BaseURL: srv.URL + "/",
		Headers: headers,
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s, srv
}

func TestNewRejectsInvalidConfig(t *testing.T) {
	t.Parallel()
	if _, err := New(Config{}); err == nil {
		t.Fatal("expected error for empty BaseURL")
	}
	if _, err := New(Config{BaseURL: "https://example.com/no-trailing-slash"}); err == nil {
		t.Fatal("expected error for BaseURL without trailing slash")
	}
	if _, err := New(Config{BaseURL: "https://example.com/"}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestGetPresentAndAbsent(t *testing.T) {
	t.Parallel()
	data := map[string][]byte{
		"zarr.json":    []byte(`{"zarr_format":3}`),
		"chunks/c/0/0": []byte("chunk-bytes"),
	}
	s, _ := newTestStore(t, data, nil, [2]string{})

	got, err := s.Get(context.Background(), "zarr.json")
	if err != nil {
		t.Fatalf("Get present: %v", err)
	}
	if string(got) != `{"zarr_format":3}` {
		t.Fatalf("Get returned %q", got)
	}

	_, err = s.Get(context.Background(), "missing.json")
	if err == nil {
		t.Fatal("expected error for missing key")
	}
	if !errors.Is(err, store.ErrKeyNotFound) {
		t.Fatalf("expected ErrKeyNotFound, got %v", err)
	}
}

func TestGetRangeReturnsCorrectSlice(t *testing.T) {
	t.Parallel()
	body := []byte("0123456789abcdef")
	s, _ := newTestStore(t, map[string][]byte{"blob": body}, nil, [2]string{})

	got, err := s.GetRange(context.Background(), "blob", 4, 6)
	if err != nil {
		t.Fatalf("GetRange: %v", err)
	}
	if string(got) != "456789" {
		t.Fatalf("GetRange returned %q want %q", got, "456789")
	}

	// Range past EOF yields empty slice via 416 path.
	got, err = s.GetRange(context.Background(), "blob", 1000, 4)
	if err != nil {
		t.Fatalf("GetRange past EOF: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected empty slice past EOF, got %q", got)
	}

	// Missing key.
	_, err = s.GetRange(context.Background(), "absent", 0, 4)
	if !errors.Is(err, store.ErrKeyNotFound) {
		t.Fatalf("expected ErrKeyNotFound, got %v", err)
	}
}

func TestGetRangeFallbackWhenServerIgnoresRange(t *testing.T) {
	t.Parallel()
	body := []byte("ABCDEFGHIJ")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Always return the full body as 200, ignoring Range.
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)

	s, err := New(Config{BaseURL: srv.URL + "/"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	got, err := s.GetRange(context.Background(), "x", 2, 5)
	if err != nil {
		t.Fatalf("GetRange: %v", err)
	}
	if string(got) != "CDEFG" {
		t.Fatalf("GetRange fallback returned %q want %q", got, "CDEFG")
	}
}

func TestExists(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t, map[string][]byte{"there": []byte("x")}, nil, [2]string{})

	ok, err := s.Exists(context.Background(), "there")
	if err != nil || !ok {
		t.Fatalf("Exists(there) = %v, %v", ok, err)
	}
	ok, err = s.Exists(context.Background(), "missing")
	if err != nil || ok {
		t.Fatalf("Exists(missing) = %v, %v", ok, err)
	}
}

func TestSetAndDeleteReturnReadOnly(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t, nil, nil, [2]string{})

	if err := s.Set(context.Background(), "k", []byte("v")); !errors.Is(err, ErrReadOnly) {
		t.Fatalf("Set: expected ErrReadOnly, got %v", err)
	}
	if err := s.Delete(context.Background(), "k"); !errors.Is(err, ErrReadOnly) {
		t.Fatalf("Delete: expected ErrReadOnly, got %v", err)
	}
	ok, err := s.SetIfNotExists(context.Background(), "k", []byte("v"))
	if ok || !errors.Is(err, ErrReadOnly) {
		t.Fatalf("SetIfNotExists: expected (false, ErrReadOnly), got (%v, %v)", ok, err)
	}
}

func TestListYieldsUnsupported(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t, nil, nil, [2]string{})

	count := 0
	for k, err := range s.List(context.Background(), "") {
		count++
		if k != "" {
			t.Fatalf("expected empty key, got %q", k)
		}
		if !errors.Is(err, ErrListUnsupported) {
			t.Fatalf("expected ErrListUnsupported, got %v", err)
		}
	}
	if count != 1 {
		t.Fatalf("expected exactly 1 yield, got %d", count)
	}
}

func TestRejectsDotDotKey(t *testing.T) {
	t.Parallel()
	s, _ := newTestStore(t, nil, nil, [2]string{})

	for _, key := range []string{"..", "../etc/passwd", "a/../b", "foo/.."} {
		if _, err := s.Get(context.Background(), key); err == nil {
			t.Fatalf("Get(%q): expected error for .. segment", key)
		}
		if _, err := s.GetRange(context.Background(), key, 0, 4); err == nil {
			t.Fatalf("GetRange(%q): expected error for .. segment", key)
		}
		if _, err := s.Exists(context.Background(), key); err == nil {
			t.Fatalf("Exists(%q): expected error for .. segment", key)
		}
	}
}

func TestHeadersAreApplied(t *testing.T) {
	t.Parallel()
	hdr := http.Header{}
	hdr.Set("X-Auth-Token", "secret-token")
	hdr.Set("X-Other", "value")

	data := map[string][]byte{"k": []byte("v")}
	s, _ := newTestStore(t, data, hdr, [2]string{"X-Auth-Token", "secret-token"})

	got, err := s.Get(context.Background(), "k")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if string(got) != "v" {
		t.Fatalf("Get returned %q want %q", got, "v")
	}

	// HEAD should also carry headers.
	ok, err := s.Exists(context.Background(), "k")
	if err != nil || !ok {
		t.Fatalf("Exists: ok=%v err=%v", ok, err)
	}

	// GetRange should also carry headers.
	rng, err := s.GetRange(context.Background(), "k", 0, 1)
	if err != nil {
		t.Fatalf("GetRange: %v", err)
	}
	if string(rng) != "v" {
		t.Fatalf("GetRange returned %q want %q", rng, "v")
	}
}

func TestHeadersClonedFromConfig(t *testing.T) {
	t.Parallel()
	hdr := http.Header{}
	hdr.Set("X-Token", "original")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Token") != "original" {
			http.Error(w, "header mutated", http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "ok")
	}))
	t.Cleanup(srv.Close)

	s, err := New(Config{BaseURL: srv.URL + "/", Headers: hdr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// Mutate caller's map after construction; should not affect store.
	hdr.Set("X-Token", "tampered")

	if _, err := s.Get(context.Background(), "k"); err != nil {
		t.Fatalf("Get: %v", err)
	}
}

func TestUnexpectedStatusIsError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	s, err := New(Config{BaseURL: srv.URL + "/"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := s.Get(context.Background(), "k"); err == nil {
		t.Fatal("expected error for 500")
	}
	if _, err := s.GetRange(context.Background(), "k", 0, 4); err == nil {
		t.Fatal("expected error for 500")
	}
	if _, err := s.Exists(context.Background(), "k"); err == nil {
		t.Fatal("expected error for 500")
	}
}

// endlessPartialContentServer answers every request with 206 and then streams
// bytes until a write fails (i.e. the client hangs up), modeling a
// misbehaving origin or proxy that ignores the requested range length.
func endlessPartialContentServer(t *testing.T, contentRange string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Range", contentRange)
		w.WriteHeader(http.StatusPartialContent)
		chunk := make([]byte, 32<<10)
		for {
			if _, err := w.Write(chunk); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestGetRangeBoundsPartialContentBody(t *testing.T) {
	t.Parallel()
	srv := endlessPartialContentServer(t, "bytes 0-15/*")
	s, err := New(Config{BaseURL: srv.URL + "/"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// Without a bound, GetRange keeps reading until the deadline fires.
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	got, err := s.GetRange(ctx, "k", 0, 16)
	if err != nil {
		t.Fatalf("GetRange: %v", err)
	}
	if len(got) != 16 {
		t.Fatalf("GetRange len = %d, want 16", len(got))
	}
}

func TestGetSuffixBoundsPartialContentBody(t *testing.T) {
	t.Parallel()
	srv := endlessPartialContentServer(t, "bytes 0-15/16")
	s, err := New(Config{BaseURL: srv.URL + "/"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	got, err := s.GetSuffix(ctx, "k", 16)
	if err != nil {
		t.Fatalf("GetSuffix: %v", err)
	}
	if len(got) != 16 {
		t.Fatalf("GetSuffix len = %d, want 16", len(got))
	}
}

func TestErrorsRedactBaseURLPassword(t *testing.T) {
	t.Parallel()
	const password = "s3cret-pw"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)
	base := strings.Replace(srv.URL, "http://", "http://user:"+password+"@", 1) + "/"

	checkRedacted := func(op string, err error) {
		t.Helper()
		if err == nil {
			t.Fatalf("%s: expected error", op)
		}
		if strings.Contains(err.Error(), password) {
			t.Errorf("%s: error leaks password: %v", op, err)
		}
	}

	_, err := New(Config{BaseURL: strings.TrimSuffix(base, "/")})
	checkRedacted("New (no trailing slash)", err)
	_, err = New(Config{BaseURL: "http://user:" + password + "@host:badport/"})
	checkRedacted("New (unparseable)", err)

	s, err := New(Config{BaseURL: base})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx := context.Background()
	_, err = s.Get(ctx, "k")
	checkRedacted("Get", err)
	_, err = s.GetRange(ctx, "k", 0, 4)
	checkRedacted("GetRange", err)
	_, err = s.GetSuffix(ctx, "k", 4)
	checkRedacted("GetSuffix", err)
	_, err = s.Exists(ctx, "k")
	checkRedacted("Exists", err)
}

// TestErrorResponsesReuseConnection checks that non-2xx bodies are drained
// before Close so the keep-alive connection goes back to the pool instead of
// being torn down.
func TestErrorResponsesReuseConnection(t *testing.T) {
	t.Parallel()
	var newConns atomic.Int32
	srv := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/fail":
			http.Error(w, "boom", http.StatusInternalServerError)
		default:
			http.NotFound(w, r)
		}
	}))
	srv.Config.ConnState = func(_ net.Conn, st http.ConnState) {
		if st == http.StateNew {
			newConns.Add(1)
		}
	}
	srv.Start()
	t.Cleanup(srv.Close)

	s, err := New(Config{BaseURL: srv.URL + "/", Client: srv.Client()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx := context.Background()
	for i := 0; i < 2; i++ {
		if _, err := s.Get(ctx, "missing"); !errors.Is(err, store.ErrKeyNotFound) {
			t.Fatalf("Get #%d: got %v, want ErrKeyNotFound", i, err)
		}
	}
	if _, err := s.GetRange(ctx, "missing", 0, 4); !errors.Is(err, store.ErrKeyNotFound) {
		t.Fatalf("GetRange: got %v, want ErrKeyNotFound", err)
	}
	if _, err := s.Get(ctx, "fail"); err == nil {
		t.Fatal("Get fail: expected error for 500")
	}
	if _, err := s.GetSuffix(ctx, "fail", 4); err == nil {
		t.Fatal("GetSuffix fail: expected error for 500")
	}
	if got := newConns.Load(); got != 1 {
		t.Fatalf("server saw %d new connections, want 1 (keep-alive reuse)", got)
	}
}
