package s3

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

var (
	closedRangeRE = regexp.MustCompile(`^bytes=(\d+)-(\d+)$`)
	suffixRangeRE = regexp.MustCompile(`^bytes=-(\d+)$`)
)

// fakeServer is an in-memory, credential-free fake of the read side of the
// S3 object API, just deep enough for Get, GetRange, GetSuffix and Exists:
// path-style GET (whole object, a closed "bytes=start-end" or a suffix
// "bytes=-n" Range) and HEAD.
//
// An object's name is the request path minus its first segment (the
// bucket). Like a strict object store, a Range that is neither form is a
// 400, and a closed range starting at or past the object's size, or a zero
// suffix, is a 416; a suffix longer than the object serves all of it.
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

// Put stores content under the object name.
func (s *fakeServer) Put(name string, content []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.objects[name] = append([]byte(nil), content...)
}

// Ranges returns the Range header of every request received so far, in
// order ("" for a request without one).
func (s *fakeServer) Ranges() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.ranges...)
}

func (s *fakeServer) serve(w http.ResponseWriter, r *http.Request) {
	rng := r.Header.Get("Range")
	_, name, _ := strings.Cut(strings.TrimPrefix(r.URL.Path, "/"), "/")
	s.mu.Lock()
	s.ranges = append(s.ranges, rng)
	content, ok := s.objects[name]
	s.mu.Unlock()

	if !ok {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	size := int64(len(content))
	w.Header().Set("Content-Type", "application/octet-stream")
	if r.Method == http.MethodHead || rng == "" {
		w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
		w.WriteHeader(http.StatusOK)
		if r.Method != http.MethodHead {
			_, _ = w.Write(content)
		}
		return
	}

	var start, end int64 // inclusive
	if m := closedRangeRE.FindStringSubmatch(rng); m != nil {
		var err1, err2 error
		start, err1 = strconv.ParseInt(m[1], 10, 64)
		end, err2 = strconv.ParseInt(m[2], 10, 64)
		if err1 != nil || err2 != nil || end < start {
			http.Error(w, fmt.Sprintf("malformed range %q", rng), http.StatusBadRequest)
			return
		}
		if start >= size {
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return
		}
		end = min(end, size-1)
	} else if m := suffixRangeRE.FindStringSubmatch(rng); m != nil {
		n, err := strconv.ParseInt(m[1], 10, 64)
		if err != nil {
			http.Error(w, fmt.Sprintf("malformed range %q", rng), http.StatusBadRequest)
			return
		}
		if n == 0 || size == 0 {
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return
		}
		start, end = max(size-n, 0), size-1
	} else {
		http.Error(w, fmt.Sprintf("malformed range %q", rng), http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, size))
	w.Header().Set("Content-Length", strconv.FormatInt(end-start+1, 10))
	w.WriteHeader(http.StatusPartialContent)
	_, _ = w.Write(content[start : end+1])
}
