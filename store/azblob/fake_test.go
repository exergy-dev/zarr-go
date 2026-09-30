package azblob

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
)

var closedRangeRE = regexp.MustCompile(`^bytes=(\d+)-(\d+)$`)

// fakeServer is an in-memory, credential-free fake of the read side of the
// Azure Blob API, just deep enough for Get, GetRange, GetSuffix and Exists:
// GET (whole blob or a closed "bytes=start-end" x-ms-range) and HEAD
// (GetProperties). Azure has no suffix range, so neither does the fake.
//
// A blob's name is the request path minus its first segment (the
// container). Like the real service, an x-ms-range that is not a closed
// range is a 400 and one starting at or past the blob's size is a 416.
// Errors carry x-ms-error-code (BlobNotFound for a 404, InvalidRange for a
// 416), which the SDK's bloberror matching reads.
type fakeServer struct {
	URL string

	mu      sync.Mutex
	objects map[string][]byte
	ranges  []string
}

// newFakeServer starts a fakeServer, closed via t.Cleanup.
func newFakeServer(t testing.TB) *fakeServer {
	t.Helper()
	s := &fakeServer{objects: map[string][]byte{}}
	srv := httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(srv.Close)
	s.URL = srv.URL
	return s
}

// Put stores content under the blob name.
func (s *fakeServer) Put(name string, content []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.objects[name] = append([]byte(nil), content...)
}

// Ranges returns the x-ms-range header of every request received so far,
// in order ("" for a request without one).
func (s *fakeServer) Ranges() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.ranges...)
}

func (s *fakeServer) serve(w http.ResponseWriter, r *http.Request) {
	rng := r.Header.Get("x-ms-range")
	_, name, _ := strings.Cut(strings.TrimPrefix(r.URL.Path, "/"), "/")
	s.mu.Lock()
	s.ranges = append(s.ranges, rng)
	content, ok := s.objects[name]
	s.mu.Unlock()

	if !ok {
		w.Header().Set("x-ms-error-code", "BlobNotFound")
		w.WriteHeader(http.StatusNotFound)
		return
	}
	size := int64(len(content))
	w.Header().Set("ETag", `"0x1"`)
	w.Header().Set("Last-Modified", "Mon, 02 Jan 2006 15:04:05 GMT")
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("x-ms-blob-type", "BlockBlob")
	if r.Method == http.MethodHead || rng == "" {
		w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
		w.WriteHeader(http.StatusOK)
		if r.Method != http.MethodHead {
			_, _ = w.Write(content)
		}
		return
	}

	m := closedRangeRE.FindStringSubmatch(rng)
	if m == nil {
		http.Error(w, fmt.Sprintf("malformed range %q", rng), http.StatusBadRequest)
		return
	}
	start, err1 := strconv.ParseInt(m[1], 10, 64)
	end, err2 := strconv.ParseInt(m[2], 10, 64) // inclusive
	if err1 != nil || err2 != nil || end < start {
		http.Error(w, fmt.Sprintf("malformed range %q", rng), http.StatusBadRequest)
		return
	}
	if start >= size {
		w.Header().Set("x-ms-error-code", "InvalidRange")
		w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
		return
	}
	end = min(end, size-1)
	w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, size))
	w.Header().Set("Content-Length", strconv.FormatInt(end-start+1, 10))
	w.WriteHeader(http.StatusPartialContent)
	_, _ = w.Write(content[start : end+1])
}
