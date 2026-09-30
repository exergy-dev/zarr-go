// Package s3 provides an S3-backed implementation of store.Store.
//
// It targets AWS S3 and S3-compatible endpoints (R2, MinIO, etc.). Credentials
// and region are resolved through the default AWS config chain unless
// explicitly overridden via Config.
package s3

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	awss3 "github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
	smithyhttp "github.com/aws/smithy-go/transport/http"

	"github.com/exergy-dev/zarr-go/internal/cloudstore"
	zstore "github.com/exergy-dev/zarr-go/store"
)

// Config configures a Store.
type Config struct {
	// Bucket is the S3 bucket name. Required.
	Bucket string
	// Prefix is an optional key prefix joined with "/" before user keys.
	Prefix string
	// Region is the AWS region. Optional; falls back to the default chain.
	Region string
	// Endpoint overrides the S3 endpoint URL. Use for R2/MinIO/etc.
	Endpoint string
	// PathStyle toggles path-style addressing on the S3 client. Most
	// non-AWS S3-compatible endpoints require true.
	PathStyle bool
}

// Store is an S3-backed implementation of store.Store.
type Store struct {
	client *awss3.Client
	bucket string
	prefix string // normalized: no leading/trailing slashes
}

// New constructs a Store from the given Config.
func New(ctx context.Context, cfg Config) (*Store, error) {
	if cfg.Bucket == "" {
		return nil, errors.New("s3: Bucket is required")
	}

	loadOpts := []func(*config.LoadOptions) error{}
	if cfg.Region != "" {
		loadOpts = append(loadOpts, config.WithRegion(cfg.Region))
	}
	awsCfg, err := config.LoadDefaultConfig(ctx, loadOpts...)
	if err != nil {
		return nil, fmt.Errorf("s3: load aws config: %w", err)
	}

	client := awss3.NewFromConfig(awsCfg, func(o *awss3.Options) {
		if cfg.Endpoint != "" {
			o.BaseEndpoint = aws.String(cfg.Endpoint)
			if cfg.PathStyle {
				o.UsePathStyle = true
			}
		} else if cfg.PathStyle {
			o.UsePathStyle = true
		}
	})

	return &Store{
		client: client,
		bucket: cfg.Bucket,
		prefix: strings.Trim(cfg.Prefix, "/"),
	}, nil
}

// isNotFound reports whether err corresponds to an S3 NoSuchKey/NotFound or a
// 404 HTTP response.
func isNotFound(err error) bool {
	if err == nil {
		return false
	}
	var nsk *types.NoSuchKey
	if errors.As(err, &nsk) {
		return true
	}
	var nf *types.NotFound
	if errors.As(err, &nf) {
		return true
	}
	return responseStatus(err) == http.StatusNotFound
}

// isInvalidRange reports whether err is a 416 Requested Range Not
// Satisfiable, returned when off is at or past the object's size.
func isInvalidRange(err error) bool {
	return responseStatus(err) == http.StatusRequestedRangeNotSatisfiable
}

// responseStatus extracts the HTTP status code from a smithy ResponseError,
// or 0 if err isn't one.
func responseStatus(err error) int {
	if err == nil {
		return 0
	}
	var re *smithyhttp.ResponseError
	if errors.As(err, &re) {
		return re.HTTPStatusCode()
	}
	return 0
}

// notFoundErr wraps store.ErrKeyNotFound with context about the missing key
// and, when available, the underlying SDK error (for debugging; err is
// still reachable via errors.Is/As through the multi-%w chain).
func notFoundErr(key string, err error) error {
	if err == nil {
		return fmt.Errorf("s3: key %q: %w", key, zstore.ErrKeyNotFound)
	}
	return fmt.Errorf("s3: key %q: %w: %w", key, zstore.ErrKeyNotFound, err)
}

// headExists implements the existence check shared by Exists and the n==0
// GetRange/GetSuffix cases. k is the already-prefix-joined S3 key.
func (s *Store) headExists(ctx context.Context, k string) (bool, error) {
	_, err := s.client.HeadObject(ctx, &awss3.HeadObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(k),
	})
	if err != nil {
		if isNotFound(err) {
			return false, nil
		}
		return false, fmt.Errorf("s3: HeadObject %q: %w", k, err)
	}
	return true, nil
}

// Get implements store.Store.
func (s *Store) Get(ctx context.Context, key string) ([]byte, error) {
	k, err := cloudstore.JoinKey("s3", s.prefix, key)
	if err != nil {
		return nil, err
	}
	out, err := s.client.GetObject(ctx, &awss3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(k),
	})
	if err != nil {
		if isNotFound(err) {
			return nil, notFoundErr(key, err)
		}
		return nil, fmt.Errorf("s3: GetObject %q: %w", k, err)
	}
	defer out.Body.Close()
	b, err := io.ReadAll(out.Body)
	if err != nil {
		return nil, fmt.Errorf("s3: read body %q: %w", k, err)
	}
	return b, nil
}

// GetRange implements store.Store, per the store.Store GetRange contract.
func (s *Store) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	if err := cloudstore.ValidateRange("s3", "GetRange", off, n); err != nil {
		return nil, err
	}
	n = cloudstore.ClampLen(off, n)
	k, err := cloudstore.JoinKey("s3", s.prefix, key)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		ok, err := s.headExists(ctx, k)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, notFoundErr(key, nil)
		}
		return []byte{}, nil
	}
	rng := fmt.Sprintf("bytes=%d-%d", off, off+n-1)
	out, err := s.client.GetObject(ctx, &awss3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(k),
		Range:  aws.String(rng),
	})
	if err != nil {
		if isNotFound(err) {
			return nil, notFoundErr(key, err)
		}
		if isInvalidRange(err) {
			// off is at or past the object's size (or the object is empty):
			// per the GetRange contract this is an empty result, not an
			// error, as long as the key exists.
			ok, hErr := s.headExists(ctx, k)
			if hErr != nil {
				return nil, hErr
			}
			if !ok {
				return nil, notFoundErr(key, nil)
			}
			return []byte{}, nil
		}
		return nil, fmt.Errorf("s3: GetObject %q range %s: %w", k, rng, err)
	}
	defer out.Body.Close()
	b, err := io.ReadAll(out.Body)
	if err != nil {
		return nil, fmt.Errorf("s3: read body %q: %w", k, err)
	}
	return b, nil
}

// GetSuffix implements store.SuffixReader using S3's native suffix-range
// syntax ("bytes=-N"), per the store.SuffixReader contract.
func (s *Store) GetSuffix(ctx context.Context, key string, n int64) ([]byte, error) {
	if err := cloudstore.ValidateRange("s3", "GetSuffix", 0, n); err != nil {
		return nil, err
	}
	k, err := cloudstore.JoinKey("s3", s.prefix, key)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		ok, err := s.headExists(ctx, k)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, notFoundErr(key, nil)
		}
		return []byte{}, nil
	}
	rng := fmt.Sprintf("bytes=-%d", n)
	out, err := s.client.GetObject(ctx, &awss3.GetObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(k),
		Range:  aws.String(rng),
	})
	if err != nil {
		if isNotFound(err) {
			return nil, notFoundErr(key, err)
		}
		if isInvalidRange(err) {
			// Some S3-compatible implementations reject a suffix range
			// longer than the object itself instead of clamping it; a
			// whole-object fetch is exactly the clamped result the
			// contract calls for.
			return s.Get(ctx, key)
		}
		return nil, fmt.Errorf("s3: GetObject %q suffix range %s: %w", k, rng, err)
	}
	defer out.Body.Close()
	b, err := io.ReadAll(out.Body)
	if err != nil {
		return nil, fmt.Errorf("s3: read body %q: %w", k, err)
	}
	return b, nil
}

// Set implements store.Store.
func (s *Store) Set(ctx context.Context, key string, value []byte) error {
	k, err := cloudstore.JoinKey("s3", s.prefix, key)
	if err != nil {
		return err
	}
	_, err = s.client.PutObject(ctx, &awss3.PutObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(k),
		Body:   bytes.NewReader(value),
	})
	if err != nil {
		return fmt.Errorf("s3: PutObject %q: %w", k, err)
	}
	return nil
}

// SetIfNotExists implements store.WritableStore using S3's If-None-Match
// conditional write (generally available since November 2024).
//
// If the object already exists S3 responds with 412 PreconditionFailed
// (older/most gateways) or, per S3's newer conditional-write documentation,
// a 409 Conflict with error code ConditionalRequestConflict when two
// If-None-Match PUTs race in flight; both translate into (false, nil) so
// callers can race safely. Any other error surfaces as (false, err). Callers
// behind older S3-compatible gateways that do not understand If-None-Match
// will see those errors here rather than silent overwrites.
func (s *Store) SetIfNotExists(ctx context.Context, key string, value []byte) (bool, error) {
	k, err := cloudstore.JoinKey("s3", s.prefix, key)
	if err != nil {
		return false, err
	}
	_, err = s.client.PutObject(ctx, &awss3.PutObjectInput{
		Bucket:      aws.String(s.bucket),
		Key:         aws.String(k),
		Body:        bytes.NewReader(value),
		IfNoneMatch: aws.String("*"),
	})
	if err != nil {
		status := responseStatus(err)
		if status == http.StatusPreconditionFailed || status == http.StatusConflict {
			return false, nil
		}
		return false, fmt.Errorf("s3: PutObject %q (if-none-match): %w", k, err)
	}
	return true, nil
}

// Delete implements store.Store. Missing keys are not an error.
func (s *Store) Delete(ctx context.Context, key string) error {
	k, err := cloudstore.JoinKey("s3", s.prefix, key)
	if err != nil {
		return err
	}
	_, err = s.client.DeleteObject(ctx, &awss3.DeleteObjectInput{
		Bucket: aws.String(s.bucket),
		Key:    aws.String(k),
	})
	if err != nil {
		if isNotFound(err) {
			return nil
		}
		return fmt.Errorf("s3: DeleteObject %q: %w", k, err)
	}
	return nil
}

// Exists implements store.Store.
func (s *Store) Exists(ctx context.Context, key string) (bool, error) {
	k, err := cloudstore.JoinKey("s3", s.prefix, key)
	if err != nil {
		return false, err
	}
	return s.headExists(ctx, k)
}

// List implements store.Store. The supplied prefix is joined with the
// store's configured prefix; yielded keys have the configured prefix
// stripped.
func (s *Store) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	full := cloudstore.ListPrefix(s.prefix, prefix)
	return func(yield func(string, error) bool) {
		paginator := awss3.NewListObjectsV2Paginator(s.client, &awss3.ListObjectsV2Input{
			Bucket: aws.String(s.bucket),
			Prefix: aws.String(full),
		})
		for paginator.HasMorePages() {
			if err := ctx.Err(); err != nil {
				yield("", err)
				return
			}
			page, err := paginator.NextPage(ctx)
			if err != nil {
				yield("", fmt.Errorf("s3: ListObjectsV2: %w", err))
				return
			}
			for _, obj := range page.Contents {
				if obj.Key == nil {
					continue
				}
				k, ok := cloudstore.KeyFromObject(s.prefix, *obj.Key)
				if !ok {
					continue
				}
				if !yield(k, nil) {
					return
				}
			}
		}
	}
}

// Compile-time checks that *Store satisfies the store interfaces.
var (
	_ zstore.Store         = (*Store)(nil)
	_ zstore.WritableStore = (*Store)(nil)
	_ zstore.SuffixReader  = (*Store)(nil)
)
