// Package http provides a read-only implementation of store.Store backed by
// HTTP GET/HEAD requests.
//
// It is intended for serving zarr datasets from static web servers, object
// stores fronted by HTTP (S3 website endpoints, GCS public buckets, etc.), or
// any other URL-addressable origin. *Store is not a store.Writer, so the
// zarr write paths reject it with store.ErrReadOnly.
//
// Listing is not supported because HTTP has no portable directory-listing
// protocol; consumers that require listing should rely on consolidated zarr
// metadata. List therefore yields a single (string, error) pair where the
// error is ErrListUnsupported.
package http

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/exergy-dev/zarr-go/store"
)

// ErrListUnsupported is returned (via the iterator) by List because HTTP
// listing requires consolidated metadata.
var ErrListUnsupported = errors.New("http store: list requires consolidated metadata")

// Config configures a Store.
type Config struct {
	// BaseURL is the prefix used for all key lookups. It must parse via
	// url.Parse and must end with "/".
	BaseURL string

	// Client is the HTTP client used for all requests. If nil,
	// http.DefaultClient is used.
	Client *http.Client

	// Headers are merged into every outgoing request. Use this to set auth
	// tokens, custom user agents, etc. The header map is cloned once, at
	// construction time (see New) — mutating cfg.Headers afterward has no
	// effect on the constructed Store; build a new Store if headers need to
	// change (e.g. to rotate a token).
	Headers http.Header
}

// Store is a read-only HTTP-backed store.Store implementation.
//
// The zero value is not usable; construct one with New.
type Store struct {
	baseURL *url.URL
	client  *http.Client
	headers http.Header
}

// New constructs a Store from cfg. It validates that BaseURL parses and ends
// with "/".
func New(cfg Config) (*Store, error) {
	if cfg.BaseURL == "" {
		return nil, errors.New("http store: BaseURL is required")
	}
	base, err := url.Parse(cfg.BaseURL)
	if err != nil {
		// Unwrap the *url.Error: its message quotes the raw URL, which
		// may include a password.
		return nil, fmt.Errorf("http store: invalid BaseURL: %w", errors.Unwrap(err))
	}
	if !strings.HasSuffix(cfg.BaseURL, "/") {
		// Redacted: a BaseURL may carry user:password@ credentials, which
		// must not end up in error strings or logs.
		return nil, fmt.Errorf("http store: BaseURL %q must end with %q", base.Redacted(), "/")
	}
	client := cfg.Client
	if client == nil {
		client = http.DefaultClient
	}
	var headers http.Header
	if cfg.Headers != nil {
		headers = cfg.Headers.Clone()
	}
	return &Store{
		baseURL: base,
		client:  client,
		headers: headers,
	}, nil
}

// resolve joins key onto the base URL and rejects keys containing ".."
// segments. This prevents callers from escaping the configured base.
//
// The returned URL may carry BaseURL credentials: format it in errors with
// u.Redacted(), never u.String() or %s/%v on u directly.
func (s *Store) resolve(key string) (*url.URL, error) {
	for _, seg := range strings.Split(key, "/") {
		if seg == ".." {
			return nil, fmt.Errorf("http store: key %q contains a \"..\" segment: %w", key, store.ErrInvalidKey)
		}
	}
	return s.baseURL.JoinPath(key), nil
}

// newRequest constructs an *http.Request and applies any configured headers.
func (s *Store) newRequest(ctx context.Context, method string, u *url.URL) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, u.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("http store: build request: %w", err)
	}
	for k, vs := range s.headers {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	return req, nil
}

// maxDrain caps how much of an unread response body drainAndClose discards.
// Small error bodies (404/500 pages, the tail of a chunked response) fit
// easily; anything larger is cheaper to abandon along with its connection.
const maxDrain = 4 << 10

// drainAndClose discards up to maxDrain bytes of any unread body before
// closing it. The transport only returns a connection to the keep-alive pool
// once the body has been read to EOF, so closing an unread error body would
// otherwise force a fresh connection (and TLS handshake) on the next request.
func drainAndClose(body io.ReadCloser) {
	_, _ = io.Copy(io.Discard, io.LimitReader(body, maxDrain))
	_ = body.Close()
}

// Get implements store.Store.
func (s *Store) Get(ctx context.Context, key string) ([]byte, error) {
	u, err := s.resolve(key)
	if err != nil {
		return nil, err
	}
	req, err := s.newRequest(ctx, http.MethodGet, u)
	if err != nil {
		return nil, err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http store: GET %s: %w", u.Redacted(), err)
	}
	defer drainAndClose(resp.Body)
	switch resp.StatusCode {
	case http.StatusOK:
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("http store: GET %s: read body: %w", u.Redacted(), err)
		}
		return body, nil
	case http.StatusNotFound:
		return nil, fmt.Errorf("http store: GET %s: %w", u.Redacted(), store.ErrKeyNotFound)
	default:
		return nil, fmt.Errorf("http store: GET %s: unexpected status %d", u.Redacted(), resp.StatusCode)
	}
}

// GetRange implements store.Store, per the store.Store GetRange contract. It
// issues a Range request and falls back to slicing the full body if the
// server returned 200 (ignoring the Range header).
func (s *Store) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	if off < 0 || n < 0 {
		return nil, fmt.Errorf("http store: GetRange(off=%d, n=%d): %w", off, n, store.ErrInvalidRange)
	}
	u, err := s.resolve(key)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		// Asking for zero bytes; skip building the GET request entirely,
		// but still validate the key exists so callers see consistent
		// errors.
		ok, exErr := s.Exists(ctx, key)
		if exErr != nil {
			return nil, exErr
		}
		if !ok {
			return nil, fmt.Errorf("http store: GetRange %s: %w", u.Redacted(), store.ErrKeyNotFound)
		}
		return []byte{}, nil
	}
	// Guard the Range header against int64 overflow: bytes=off-(off+n-1)
	// must be representable. Anything that would overflow is far past any
	// real object size, so cap n to whatever fits.
	if off > 0 && off > math.MaxInt64-(n-1) {
		n = math.MaxInt64 - off + 1
	}
	req, err := s.newRequest(ctx, http.MethodGet, u)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", off, off+n-1))
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http store: GET %s (range): %w", u.Redacted(), err)
	}
	defer drainAndClose(resp.Body)
	switch resp.StatusCode {
	case http.StatusPartialContent:
		// A server that honors Range is expected to report the actual
		// range served via Content-Range; if it disagrees with what we
		// asked for, the body isn't what the caller thinks it is (a proxy
		// or misconfigured origin serving the wrong bytes) — fail loudly
		// rather than silently returning mismatched data.
		if cr := resp.Header.Get("Content-Range"); cr != "" {
			if start, ok := parseContentRangeStart(cr); ok && start != off {
				return nil, fmt.Errorf("http store: GET %s (range): server returned Content-Range %q for requested offset %d", u.Redacted(), cr, off)
			}
		}
		// Bound the read at n: a misbehaving server streaming more than
		// the requested range (or an endless body) must not grow memory
		// without limit.
		body, err := io.ReadAll(io.LimitReader(resp.Body, n))
		if err != nil {
			return nil, fmt.Errorf("http store: GET %s (range): read body: %w", u.Redacted(), err)
		}
		return body, nil
	case http.StatusOK:
		// Server ignored the Range header; slice manually.
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("http store: GET %s (range fallback): read body: %w", u.Redacted(), err)
		}
		clamped := store.ClampRange(body, off, n)
		out := make([]byte, len(clamped))
		copy(out, clamped)
		return out, nil
	case http.StatusNotFound:
		return nil, fmt.Errorf("http store: GET %s (range): %w", u.Redacted(), store.ErrKeyNotFound)
	case http.StatusRequestedRangeNotSatisfiable:
		// Treat as empty slice; the key might exist but the requested range
		// is past EOF. Return a zero-length slice rather than an error so
		// callers reading optional trailing bytes don't see a failure.
		return []byte{}, nil
	default:
		return nil, fmt.Errorf("http store: GET %s (range): unexpected status %d", u.Redacted(), resp.StatusCode)
	}
}

// GetSuffix implements store.SuffixReader using an HTTP suffix range
// request (Range: bytes=-N). Falls back to a whole-body fetch and manual
// slice if the server returns 200 (ignoring Range) or 416 (rejecting a
// suffix longer than the resource).
func (s *Store) GetSuffix(ctx context.Context, key string, n int64) ([]byte, error) {
	if n < 0 {
		return nil, fmt.Errorf("http store: GetSuffix(n=%d): %w", n, store.ErrInvalidRange)
	}
	u, err := s.resolve(key)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		ok, exErr := s.Exists(ctx, key)
		if exErr != nil {
			return nil, exErr
		}
		if !ok {
			return nil, fmt.Errorf("http store: GetSuffix %s: %w", u.Redacted(), store.ErrKeyNotFound)
		}
		return []byte{}, nil
	}
	req, err := s.newRequest(ctx, http.MethodGet, u)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Range", fmt.Sprintf("bytes=-%d", n))
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http store: GET %s (suffix range): %w", u.Redacted(), err)
	}
	defer drainAndClose(resp.Body)
	switch resp.StatusCode {
	case http.StatusPartialContent:
		// Bound the read at n, as in GetRange: a compliant server sends
		// exactly the suffix, and anything beyond n is not trusted.
		body, err := io.ReadAll(io.LimitReader(resp.Body, n))
		if err != nil {
			return nil, fmt.Errorf("http store: GET %s (suffix range): read body: %w", u.Redacted(), err)
		}
		return body, nil
	case http.StatusOK:
		// Server ignored the Range header; slice manually.
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("http store: GET %s (suffix range fallback): read body: %w", u.Redacted(), err)
		}
		length := int64(len(body))
		start := length - n
		if start < 0 {
			start = 0
		}
		return body[start:], nil
	case http.StatusNotFound:
		return nil, fmt.Errorf("http store: GET %s (suffix range): %w", u.Redacted(), store.ErrKeyNotFound)
	case http.StatusRequestedRangeNotSatisfiable:
		// n is at or past the resource's size; the contract calls for the
		// whole value in that case.
		return s.Get(ctx, key)
	default:
		return nil, fmt.Errorf("http store: GET %s (suffix range): unexpected status %d", u.Redacted(), resp.StatusCode)
	}
}

// parseContentRangeStart extracts the start offset from a Content-Range
// header value of the form "bytes <start>-<end>/<size-or-*>". ok is false
// if the header isn't in the expected form.
func parseContentRangeStart(h string) (start int64, ok bool) {
	const prefix = "bytes "
	if !strings.HasPrefix(h, prefix) {
		return 0, false
	}
	rest := h[len(prefix):]
	dash := strings.IndexByte(rest, '-')
	if dash < 0 {
		return 0, false
	}
	start, err := strconv.ParseInt(rest[:dash], 10, 64)
	if err != nil {
		return 0, false
	}
	return start, true
}

// Exists implements store.Store using HEAD.
func (s *Store) Exists(ctx context.Context, key string) (bool, error) {
	u, err := s.resolve(key)
	if err != nil {
		return false, err
	}
	req, err := s.newRequest(ctx, http.MethodHead, u)
	if err != nil {
		return false, err
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return false, fmt.Errorf("http store: HEAD %s: %w", u.Redacted(), err)
	}
	defer drainAndClose(resp.Body)
	switch resp.StatusCode {
	case http.StatusOK, http.StatusNoContent:
		return true, nil
	case http.StatusNotFound:
		return false, nil
	default:
		return false, fmt.Errorf("http store: HEAD %s: unexpected status %d", u.Redacted(), resp.StatusCode)
	}
}

// List yields a single (string, error) pair where the error is
// ErrListUnsupported. HTTP has no portable directory-listing protocol; use a
// consolidated metadata file instead.
func (s *Store) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		yield("", ErrListUnsupported)
	}
}

// Compile-time interface assertions.
var (
	_ store.Store        = (*Store)(nil)
	_ store.SuffixReader = (*Store)(nil)
)
