// Package cloudstoretest provides credential-free httptest fakes for the
// s3, gcs, and azblob backends' tests.
package cloudstoretest

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"sync"
	"testing"
)

var rangeRE = regexp.MustCompile(`^bytes=(\d+)-(\d+)$`)

// RangeServer starts an httptest server that serves content for any path,
// honoring a closed byte range ("bytes=start-end") read from the header
// named rangeHeader ("Range" for S3/GCS, "x-ms-range" for Azure). Like a
// strict object store, it answers 400 to a range header that is not two
// non-negative decimal offsets with start <= end, and 416 when start is at
// or past len(content). ranges returns every range header value received,
// in order. The server is closed via t.Cleanup.
func RangeServer(t testing.TB, rangeHeader string, content []byte) (url string, ranges func() []string) {
	t.Helper()
	var (
		mu  sync.Mutex
		got []string
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rng := r.Header.Get(rangeHeader)
		mu.Lock()
		got = append(got, rng)
		mu.Unlock()

		size := int64(len(content))
		w.Header().Set("ETag", `"0x1"`)
		w.Header().Set("Last-Modified", "Mon, 02 Jan 2006 15:04:05 GMT")
		w.Header().Set("Content-Type", "application/octet-stream")
		m := rangeRE.FindStringSubmatch(rng)
		if m == nil {
			http.Error(w, fmt.Sprintf("malformed range %q", rng), http.StatusBadRequest)
			return
		}
		start, err1 := strconv.ParseInt(m[1], 10, 64)
		end, err2 := strconv.ParseInt(m[2], 10, 64)
		if err1 != nil || err2 != nil || end < start {
			http.Error(w, fmt.Sprintf("malformed range %q", rng), http.StatusBadRequest)
			return
		}
		if start >= size {
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return
		}
		end = min(end, size-1)
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, size))
		w.Header().Set("Content-Length", strconv.FormatInt(end-start+1, 10))
		w.WriteHeader(http.StatusPartialContent)
		_, _ = w.Write(content[start : end+1])
	}))
	t.Cleanup(srv.Close)
	return srv.URL, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), got...)
	}
}
