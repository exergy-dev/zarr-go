// Package cloudstoretest provides a credential-free httptest fake for the
// s3, gcs, and azblob backends' tests.
package cloudstoretest

import (
	"encoding/json"
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

// Server is an in-memory fake of the read side of the S3, GCS and Azure
// Blob object APIs, just deep enough for the backends' Get, GetRange,
// GetSuffix and Exists: GET (whole object, a closed "bytes=start-end" or a
// suffix "bytes=-n" range), HEAD, and GCS's JSON object-metadata GET.
//
// An object's name is the request path minus its first segment (the bucket
// or container); GCS metadata requests use /storage/v1/b/<bucket>/o/<name>.
// Like a strict object store, a range header that is neither form is a
// 400, and a closed range starting at or past the object's size, or a zero
// suffix, is a 416; a suffix longer than the object serves all of it.
// Errors carry Azure's x-ms-error-code (BlobNotFound for a 404, InvalidRange
// for a 416).
type Server struct {
	URL string

	rangeHeader string

	mu      sync.Mutex
	objects map[string][]byte
	ranges  []string
}

// NewServer starts a Server that reads byte ranges from the header named
// rangeHeader ("Range" for S3/GCS, "x-ms-range" for Azure). It is closed
// via t.Cleanup.
func NewServer(t testing.TB, rangeHeader string) *Server {
	t.Helper()
	s := &Server{rangeHeader: rangeHeader, objects: map[string][]byte{}}
	srv := httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(srv.Close)
	s.URL = srv.URL
	return s
}

// Put stores content under the object name.
func (s *Server) Put(name string, content []byte) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.objects[name] = append([]byte(nil), content...)
}

// Ranges returns the range header of every request received so far, in
// order ("" for a request without one).
func (s *Server) Ranges() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.ranges...)
}

func (s *Server) serve(w http.ResponseWriter, r *http.Request) {
	rng := r.Header.Get(s.rangeHeader)
	name, gcsMeta := strings.CutPrefix(r.URL.Path, "/storage/v1/b/")
	if gcsMeta {
		_, name, _ = strings.Cut(name, "/o/")
	} else {
		_, name, _ = strings.Cut(strings.TrimPrefix(name, "/"), "/")
	}
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
	if gcsMeta {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{
			"bucket": "b", "name": name, "size": strconv.FormatInt(size, 10),
		})
		return
	}
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
			w.Header().Set("x-ms-error-code", "InvalidRange")
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
			w.Header().Set("x-ms-error-code", "InvalidRange")
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
