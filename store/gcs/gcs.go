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

	zstore "github.com/rkm/zarr-go/store"
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
		prefix: trimSlashes(cfg.Prefix),
	}, nil
}

// Close releases resources held by the underlying GCS client.
func (s *Store) Close() error {
	if s.client == nil {
		return nil
	}
	return s.client.Close()
}

// joinKey joins the configured prefix with key, producing a clean GCS object
// name with no leading or duplicate slashes.
func joinKey(prefix, key string) string {
	prefix = trimSlashes(prefix)
	key = strings.TrimLeft(key, "/")
	if prefix == "" {
		return key
	}
	if key == "" {
		return prefix
	}
	return prefix + "/" + key
}

func trimSlashes(s string) string {
	return strings.Trim(s, "/")
}

// stripPrefix removes the configured prefix from a returned GCS object name.
func stripPrefix(prefix, key string) string {
	if prefix == "" {
		return key
	}
	p := prefix + "/"
	return strings.TrimPrefix(key, p)
}

// notFoundErr wraps store.ErrKeyNotFound with context about the missing key.
func notFoundErr(key string) error {
	return fmt.Errorf("gcs: key %q: %w", key, zstore.ErrKeyNotFound)
}

// isPreconditionFailed reports whether err is a 412 PreconditionFailed from
// the GCS API (used to detect SetIfNotExists conflicts).
func isPreconditionFailed(err error) bool {
	if err == nil {
		return false
	}
	var ge *googleapi.Error
	if errors.As(err, &ge) {
		return ge.Code == 412
	}
	return false
}

// Get implements store.Store.
func (s *Store) Get(ctx context.Context, key string) ([]byte, error) {
	k := joinKey(s.prefix, key)
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

// GetRange implements store.Store.
func (s *Store) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	if n <= 0 {
		return []byte{}, nil
	}
	k := joinKey(s.prefix, key)
	r, err := s.bucket.Object(k).NewRangeReader(ctx, off, n)
	if err != nil {
		if errors.Is(err, storage.ErrObjectNotExist) {
			return nil, notFoundErr(key)
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

// Set implements store.Store.
func (s *Store) Set(ctx context.Context, key string, value []byte) error {
	k := joinKey(s.prefix, key)
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

// SetIfNotExists implements store.WritableStore. It uses an
// "x-goog-if-generation-match: 0" precondition (DoesNotExist) so concurrent
// writers can race safely; the loser receives a 412 and we return (false, nil).
func (s *Store) SetIfNotExists(ctx context.Context, key string, value []byte) (bool, error) {
	k := joinKey(s.prefix, key)
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

// Delete implements store.Store. Missing keys are not an error.
func (s *Store) Delete(ctx context.Context, key string) error {
	k := joinKey(s.prefix, key)
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
	k := joinKey(s.prefix, key)
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
	full := joinKey(s.prefix, prefix)
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
			if attrs == nil || attrs.Name == "" {
				continue
			}
			k := stripPrefix(s.prefix, attrs.Name)
			if !yield(k, nil) {
				return
			}
		}
	}
}

// Compile-time checks that *Store satisfies the store interfaces.
var (
	_ zstore.Store         = (*Store)(nil)
	_ zstore.WritableStore = (*Store)(nil)
)
