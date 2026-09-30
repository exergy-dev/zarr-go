// Package gcs provides a Google Cloud Storage-backed implementation of
// store.Store.
//
// Authentication uses Application Default Credentials via
// storage.NewClient(ctx). Configure via gcloud auth application-default login,
// GOOGLE_APPLICATION_CREDENTIALS, or workload identity.
package gcs

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"strings"

	"cloud.google.com/go/storage"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/iterator"

	"github.com/exergy-dev/zarr-go/internal/cloudstore"
	zstore "github.com/exergy-dev/zarr-go/store"
)

// Config configures a Store.
type Config struct {
	// Bucket is the GCS bucket name. Required.
	Bucket string
	// Prefix is an optional key prefix joined with "/" before user keys.
	Prefix string
}

// Store is a Google Cloud Storage-backed implementation of store.Store.
type Store struct {
	client *storage.Client
	bucket *storage.BucketHandle
	name   string
	prefix string // normalized: no leading/trailing slashes
}

// New constructs a Store from the given Config using Application Default
// Credentials.
func New(ctx context.Context, cfg Config) (*Store, error) {
	if cfg.Bucket == "" {
		return nil, errors.New("gcs: Bucket is required")
	}
	client, err := storage.NewClient(ctx)
	if err != nil {
		return nil, fmt.Errorf("gcs: new client: %w", err)
	}
	return &Store{
		client: client,
		bucket: client.Bucket(cfg.Bucket),
		name:   cfg.Bucket,
		prefix: strings.Trim(cfg.Prefix, "/"),
	}, nil
}

// Close releases resources held by the underlying GCS client.
func (s *Store) Close() error {
	if s.client == nil {
		return nil
	}
	return s.client.Close()
}

// notFoundErr wraps store.ErrKeyNotFound with context about the missing key.
func notFoundErr(key string) error {
	return fmt.Errorf("gcs: key %q: %w", key, zstore.ErrKeyNotFound)
}

// isPreconditionFailed reports whether err is a 412 PreconditionFailed from
// the GCS API (used to detect SetIfNotExists conflicts).
func isPreconditionFailed(err error) bool {
	return googleAPICode(err) == 412
}

// isInvalidRange reports whether err is a 416 Requested Range Not
// Satisfiable. Verified against fake-gcs-server that GCS only returns this
// once object existence has already been confirmed (a missing object fails
// with storage.ErrObjectNotExist before range validity is ever evaluated),
// so a 416 always means "off is at or past the object's size", not "missing
// key".
func isInvalidRange(err error) bool {
	return googleAPICode(err) == 416
}

// googleAPICode extracts the HTTP status code from a *googleapi.Error, or 0
// if err isn't one.
func googleAPICode(err error) int {
	if err == nil {
		return 0
	}
	var ge *googleapi.Error
	if errors.As(err, &ge) {
		return ge.Code
	}
	return 0
}

// Get implements store.Store.
func (s *Store) Get(ctx context.Context, key string) ([]byte, error) {
	k, err := cloudstore.JoinKey("gcs", s.prefix, key)
	if err != nil {
		return nil, err
	}
	r, err := s.bucket.Object(k).NewReader(ctx)
	if err != nil {
		if errors.Is(err, storage.ErrObjectNotExist) {
			return nil, notFoundErr(key)
		}
		return nil, fmt.Errorf("gcs: NewReader %q: %w", k, err)
	}
	defer r.Close()
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("gcs: read body %q: %w", k, err)
	}
	return b, nil
}

// GetRange implements store.Store, per the store.Store GetRange contract.
//
// off/n are passed through to NewRangeReader, which already implements the
// n==0 (empty, existence-checked via a real request) and clamp-past-EOF
// behavior the contract calls for; only the negative-off/n rejection, the
// off-at/past-EOF (416) case, and clamping n so the SDK's unguarded
// off+n-1 Range end cannot overflow need handling here.
func (s *Store) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	if err := cloudstore.ValidateRange("gcs", "GetRange", off, n); err != nil {
		return nil, err
	}
	n = cloudstore.ClampLen(off, n)
	k, err := cloudstore.JoinKey("gcs", s.prefix, key)
	if err != nil {
		return nil, err
	}
	r, err := s.bucket.Object(k).NewRangeReader(ctx, off, n)
	if err != nil {
		if errors.Is(err, storage.ErrObjectNotExist) {
			return nil, notFoundErr(key)
		}
		if isInvalidRange(err) {
			return []byte{}, nil
		}
		return nil, fmt.Errorf("gcs: NewRangeReader %q [%d,%d): %w", k, off, off+n, err)
	}
	defer r.Close()
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("gcs: read body %q: %w", k, err)
	}
	return b, nil
}

// GetSuffix implements store.Store. GCS's NewRangeReader overloads a
// negative offset to mean "read the last abs(offset) bytes" (with length
// forced to -1), which is exactly GetSuffix's semantics for n > 0 — but that
// trick can't express n==0 (there's no negative zero), so that case is
// handled separately via Attrs.
func (s *Store) GetSuffix(ctx context.Context, key string, n int64) ([]byte, error) {
	if err := cloudstore.ValidateRange("gcs", "GetSuffix", 0, n); err != nil {
		return nil, err
	}
	k, err := cloudstore.JoinKey("gcs", s.prefix, key)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		if _, err := s.bucket.Object(k).Attrs(ctx); err != nil {
			if errors.Is(err, storage.ErrObjectNotExist) {
				return nil, notFoundErr(key)
			}
			return nil, fmt.Errorf("gcs: Attrs %q: %w", k, err)
		}
		return []byte{}, nil
	}
	r, err := s.bucket.Object(k).NewRangeReader(ctx, -n, -1)
	if err != nil {
		if errors.Is(err, storage.ErrObjectNotExist) {
			return nil, notFoundErr(key)
		}
		return nil, fmt.Errorf("gcs: NewRangeReader %q suffix -%d: %w", k, n, err)
	}
	defer r.Close()
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("gcs: read body %q: %w", k, err)
	}
	return b, nil
}

// Set implements store.Writer.
func (s *Store) Set(ctx context.Context, key string, value []byte) error {
	k, err := cloudstore.JoinKey("gcs", s.prefix, key)
	if err != nil {
		return err
	}
	w := s.bucket.Object(k).NewWriter(ctx)
	if _, err := w.Write(value); err != nil {
		_ = w.Close()
		return fmt.Errorf("gcs: write %q: %w", k, err)
	}
	if err := w.Close(); err != nil {
		return fmt.Errorf("gcs: close writer %q: %w", k, err)
	}
	return nil
}

// SetIfNotExists implements store.Writer. It uses an
// "x-goog-if-generation-match: 0" precondition (DoesNotExist) so concurrent
// writers can race safely; the loser receives a 412 and we return (false, nil).
func (s *Store) SetIfNotExists(ctx context.Context, key string, value []byte) (bool, error) {
	k, err := cloudstore.JoinKey("gcs", s.prefix, key)
	if err != nil {
		return false, err
	}
	obj := s.bucket.Object(k).If(storage.Conditions{DoesNotExist: true})
	w := obj.NewWriter(ctx)
	if _, err := w.Write(value); err != nil {
		_ = w.Close()
		if isPreconditionFailed(err) {
			return false, nil
		}
		return false, fmt.Errorf("gcs: write %q: %w", k, err)
	}
	if err := w.Close(); err != nil {
		if isPreconditionFailed(err) {
			return false, nil
		}
		return false, fmt.Errorf("gcs: close writer %q: %w", k, err)
	}
	return true, nil
}

// Delete implements store.Writer. Missing keys are not an error.
func (s *Store) Delete(ctx context.Context, key string) error {
	k, err := cloudstore.JoinKey("gcs", s.prefix, key)
	if err != nil {
		return err
	}
	if err := s.bucket.Object(k).Delete(ctx); err != nil {
		if errors.Is(err, storage.ErrObjectNotExist) {
			return nil
		}
		return fmt.Errorf("gcs: Delete %q: %w", k, err)
	}
	return nil
}

// Exists implements store.Store.
func (s *Store) Exists(ctx context.Context, key string) (bool, error) {
	k, err := cloudstore.JoinKey("gcs", s.prefix, key)
	if err != nil {
		return false, err
	}
	if _, err := s.bucket.Object(k).Attrs(ctx); err != nil {
		if errors.Is(err, storage.ErrObjectNotExist) {
			return false, nil
		}
		return false, fmt.Errorf("gcs: Attrs %q: %w", k, err)
	}
	return true, nil
}

// List implements store.Store. The supplied prefix is joined with the store's
// configured prefix; yielded keys have the configured prefix stripped.
func (s *Store) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	full := cloudstore.ListPrefix(s.prefix, prefix)
	return func(yield func(string, error) bool) {
		it := s.bucket.Objects(ctx, &storage.Query{Prefix: full})
		for {
			if err := ctx.Err(); err != nil {
				yield("", err)
				return
			}
			attrs, err := it.Next()
			if errors.Is(err, iterator.Done) {
				return
			}
			if err != nil {
				yield("", fmt.Errorf("gcs: list %q: %w", full, err))
				return
			}
			if attrs == nil {
				continue
			}
			k, ok := cloudstore.KeyFromObject(s.prefix, attrs.Name)
			if !ok {
				continue
			}
			if !yield(k, nil) {
				return
			}
		}
	}
}

// Compile-time interface assertion.
var _ zstore.Writer = (*Store)(nil)
