// Package azblob provides an Azure Blob Storage-backed implementation of
// store.Store.
//
// Authentication uses a shared-key credential when Config.AccountKey is
// supplied. Other auth modes (managed identity, service principal, etc.)
// require the azidentity SDK package, which is not currently a dependency
// of this module; callers that need them must vendor azidentity themselves
// and construct the client outside this package.
package azblob

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"strings"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/to"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/blob"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/bloberror"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/blockblob"

	zstore "github.com/rkm/zarr-go/store"
)

// Config configures a Store.
type Config struct {
	// AccountName is the Azure storage account name, e.g. "mystorageaccount".
	// Required unless ServiceURL fully specifies the endpoint.
	AccountName string
	// AccountKey is the shared-key credential for the storage account. When
	// empty, this package currently has no fallback authentication and New
	// will return an error. (See package doc.)
	AccountKey string
	// Container is the blob container name. Required.
	Container string
	// Prefix is an optional key prefix joined with "/" before user keys.
	Prefix string
	// ServiceURL overrides the default service URL. When empty, the URL is
	// derived as https://<AccountName>.blob.core.windows.net/.
	ServiceURL string
}

// Store is an Azure Blob Storage-backed implementation of store.Store.
type Store struct {
	client    *azblob.Client
	container string
	prefix    string // normalized: no leading/trailing slashes
}

// New constructs a Store from the given Config.
func New(ctx context.Context, cfg Config) (*Store, error) {
	if cfg.Container == "" {
		return nil, errors.New("azblob: Container is required")
	}
	if cfg.AccountName == "" && cfg.ServiceURL == "" {
		return nil, errors.New("azblob: AccountName or ServiceURL is required")
	}

	serviceURL := cfg.ServiceURL
	if serviceURL == "" {
		serviceURL = fmt.Sprintf("https://%s.blob.core.windows.net/", cfg.AccountName)
	}

	if cfg.AccountKey == "" {
		// azidentity is not a dependency of this module; without a shared
		// key we can't authenticate.
		return nil, errors.New("azblob: AccountKey is required (azidentity-based auth not supported by this package)")
	}

	cred, err := azblob.NewSharedKeyCredential(cfg.AccountName, cfg.AccountKey)
	if err != nil {
		return nil, fmt.Errorf("azblob: shared key credential: %w", err)
	}
	client, err := azblob.NewClientWithSharedKeyCredential(serviceURL, cred, nil)
	if err != nil {
		return nil, fmt.Errorf("azblob: new client: %w", err)
	}

	return &Store{
		client:    client,
		container: cfg.Container,
		prefix:    trimSlashes(cfg.Prefix),
	}, nil
}

// joinKey joins the configured prefix with key, producing a clean blob key
// with no leading or duplicate slashes.
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

// stripPrefix removes the configured prefix from a returned blob key.
func stripPrefix(prefix, key string) string {
	if prefix == "" {
		return key
	}
	p := prefix + "/"
	return strings.TrimPrefix(key, p)
}

// isNotFound reports whether err corresponds to a BlobNotFound from the
// Azure SDK.
func isNotFound(err error) bool {
	if err == nil {
		return false
	}
	return bloberror.HasCode(err, bloberror.BlobNotFound)
}

// isPreconditionFailed reports whether err corresponds to a 412 Precondition
// Failed (used by SetIfNotExists).
func isPreconditionFailed(err error) bool {
	if err == nil {
		return false
	}
	return bloberror.HasCode(err, bloberror.ConditionNotMet) ||
		bloberror.HasCode(err, bloberror.BlobAlreadyExists)
}

// notFoundErr wraps store.ErrKeyNotFound with context about the missing key.
func notFoundErr(key string, _ error) error {
	return fmt.Errorf("azblob: key %q: %w", key, zstore.ErrKeyNotFound)
}

// Get implements store.Store.
func (s *Store) Get(ctx context.Context, key string) ([]byte, error) {
	k := joinKey(s.prefix, key)
	resp, err := s.client.DownloadStream(ctx, s.container, k, nil)
	if err != nil {
		if isNotFound(err) {
			return nil, notFoundErr(key, err)
		}
		return nil, fmt.Errorf("azblob: DownloadStream %q: %w", k, err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("azblob: read body %q: %w", k, err)
	}
	return b, nil
}

// GetRange implements store.Store.
func (s *Store) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	if n <= 0 {
		return []byte{}, nil
	}
	k := joinKey(s.prefix, key)
	resp, err := s.client.DownloadStream(ctx, s.container, k, &azblob.DownloadStreamOptions{
		Range: blob.HTTPRange{Offset: off, Count: n},
	})
	if err != nil {
		if isNotFound(err) {
			return nil, notFoundErr(key, err)
		}
		return nil, fmt.Errorf("azblob: DownloadStream %q range [%d,%d): %w", k, off, off+n, err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("azblob: read body %q: %w", k, err)
	}
	return b, nil
}

// Set implements store.Store.
func (s *Store) Set(ctx context.Context, key string, value []byte) error {
	k := joinKey(s.prefix, key)
	_, err := s.client.UploadBuffer(ctx, s.container, k, value, nil)
	if err != nil {
		return fmt.Errorf("azblob: UploadBuffer %q: %w", k, err)
	}
	return nil
}

// SetIfNotExists implements store.WritableStore. It uses an If-None-Match: *
// access condition; on 412 PreconditionFailed (or BlobAlreadyExists) the
// blob already exists and (false, nil) is returned.
func (s *Store) SetIfNotExists(ctx context.Context, key string, value []byte) (bool, error) {
	k := joinKey(s.prefix, key)
	bbClient := s.client.ServiceClient().NewContainerClient(s.container).NewBlockBlobClient(k)
	body := newReadSeekCloser(value)
	_, err := bbClient.Upload(ctx, body, &blockblob.UploadOptions{
		AccessConditions: &blob.AccessConditions{
			ModifiedAccessConditions: &blob.ModifiedAccessConditions{
				IfNoneMatch: to.Ptr(azcore.ETagAny),
			},
		},
	})
	if err != nil {
		if isPreconditionFailed(err) {
			return false, nil
		}
		return false, fmt.Errorf("azblob: Upload (if-none-match) %q: %w", k, err)
	}
	return true, nil
}

// Delete implements store.Store. Missing keys are not an error.
func (s *Store) Delete(ctx context.Context, key string) error {
	k := joinKey(s.prefix, key)
	_, err := s.client.DeleteBlob(ctx, s.container, k, nil)
	if err != nil {
		if isNotFound(err) {
			return nil
		}
		return fmt.Errorf("azblob: DeleteBlob %q: %w", k, err)
	}
	return nil
}

// Exists implements store.Store.
func (s *Store) Exists(ctx context.Context, key string) (bool, error) {
	k := joinKey(s.prefix, key)
	bbClient := s.client.ServiceClient().NewContainerClient(s.container).NewBlockBlobClient(k)
	_, err := bbClient.GetProperties(ctx, nil)
	if err != nil {
		if isNotFound(err) {
			return false, nil
		}
		return false, fmt.Errorf("azblob: GetProperties %q: %w", k, err)
	}
	return true, nil
}

// List implements store.Store. The supplied prefix is joined with the
// store's configured prefix; yielded keys have the configured prefix
// stripped.
func (s *Store) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	full := joinKey(s.prefix, prefix)
	return func(yield func(string, error) bool) {
		opts := &azblob.ListBlobsFlatOptions{}
		if full != "" {
			opts.Prefix = to.Ptr(full)
		}
		pager := s.client.NewListBlobsFlatPager(s.container, opts)
		for pager.More() {
			if err := ctx.Err(); err != nil {
				yield("", err)
				return
			}
			page, err := pager.NextPage(ctx)
			if err != nil {
				yield("", fmt.Errorf("azblob: ListBlobsFlat: %w", err))
				return
			}
			if page.Segment == nil {
				continue
			}
			for _, item := range page.Segment.BlobItems {
				if item == nil || item.Name == nil {
					continue
				}
				k := stripPrefix(s.prefix, *item.Name)
				if !yield(k, nil) {
					return
				}
			}
		}
	}
}

// readSeekCloser adapts an in-memory byte slice to io.ReadSeekCloser, which
// blockblob.Client.Upload requires.
type readSeekCloser struct {
	data []byte
	pos  int64
}

func newReadSeekCloser(b []byte) *readSeekCloser {
	return &readSeekCloser{data: b}
}

func (r *readSeekCloser) Read(p []byte) (int, error) {
	if r.pos >= int64(len(r.data)) {
		return 0, io.EOF
	}
	n := copy(p, r.data[r.pos:])
	r.pos += int64(n)
	return n, nil
}

func (r *readSeekCloser) Seek(offset int64, whence int) (int64, error) {
	var abs int64
	switch whence {
	case io.SeekStart:
		abs = offset
	case io.SeekCurrent:
		abs = r.pos + offset
	case io.SeekEnd:
		abs = int64(len(r.data)) + offset
	default:
		return 0, errors.New("azblob: invalid whence")
	}
	if abs < 0 {
		return 0, errors.New("azblob: negative seek position")
	}
	r.pos = abs
	return abs, nil
}

func (r *readSeekCloser) Close() error { return nil }

// Compile-time checks that *Store satisfies the store interfaces.
var (
	_ zstore.Store         = (*Store)(nil)
	_ zstore.WritableStore = (*Store)(nil)
)
