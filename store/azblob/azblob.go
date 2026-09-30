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
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"iter"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/streaming"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/to"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/blob"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/bloberror"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/blockblob"

	"github.com/exergy-dev/zarr-go/internal/cloudstore"
	zstore "github.com/exergy-dev/zarr-go/store"
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
		prefix:    cloudstore.TrimSlashes(cfg.Prefix),
	}, nil
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

// invalidRangeCode is the Azure Blob Storage error code for a 416 Requested
// Range Not Satisfiable. bloberror has no predefined constant for it, but
// bloberror.Code is just a string type, so this is equally valid input to
// bloberror.HasCode.
const invalidRangeCode bloberror.Code = "InvalidRange"

// isInvalidRange reports whether err is a 416 Requested Range Not
// Satisfiable, returned when off is past the blob's size.
func isInvalidRange(err error) bool {
	if err == nil {
		return false
	}
	return bloberror.HasCode(err, invalidRangeCode)
}

// notFoundErr wraps store.ErrKeyNotFound with context about the missing key.
func notFoundErr(key string, _ error) error {
	return fmt.Errorf("azblob: key %q: %w", key, zstore.ErrKeyNotFound)
}

// blobExists implements the existence check shared by Exists and the n==0
// GetRange case. k is the already-prefix-joined blob key.
func (s *Store) blobExists(ctx context.Context, k string) (bool, error) {
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

// Get implements store.Store.
func (s *Store) Get(ctx context.Context, key string) ([]byte, error) {
	k := cloudstore.JoinKey(s.prefix, key)
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

// GetRange implements store.Store, per the store.Store GetRange contract.
//
// n==0 is handled as an explicit existence check rather than passed through
// to blob.HTTPRange: Count:0 in that struct means "read to the end of the
// blob" (a zero-value HTTPRange field is a "not specified" sentinel, not a
// literal zero-length range), so passing n==0 straight through would
// silently return the ENTIRE blob instead of an empty slice.
func (s *Store) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	if err := cloudstore.ValidateRange("azblob", "GetRange", off, n); err != nil {
		return nil, err
	}
	n = cloudstore.ClampLen(off, n)
	k := cloudstore.JoinKey(s.prefix, key)
	if n == 0 {
		ok, err := s.blobExists(ctx, k)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, notFoundErr(key, nil)
		}
		return []byte{}, nil
	}
	resp, err := s.client.DownloadStream(ctx, s.container, k, &azblob.DownloadStreamOptions{
		Range: blob.HTTPRange{Offset: off, Count: n},
	})
	if err != nil {
		if isNotFound(err) {
			return nil, notFoundErr(key, err)
		}
		if isInvalidRange(err) {
			return []byte{}, nil
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
	k := cloudstore.JoinKey(s.prefix, key)
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
	k := cloudstore.JoinKey(s.prefix, key)
	bbClient := s.client.ServiceClient().NewContainerClient(s.container).NewBlockBlobClient(k)
	body := streaming.NopCloser(bytes.NewReader(value))
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
	k := cloudstore.JoinKey(s.prefix, key)
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
	return s.blobExists(ctx, cloudstore.JoinKey(s.prefix, key))
}

// List implements store.Store. The supplied prefix is joined with the
// store's configured prefix; yielded keys have the configured prefix
// stripped.
func (s *Store) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	full := cloudstore.ListPrefix(s.prefix, prefix)
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
				k, ok := cloudstore.KeyFromObject(s.prefix, *item.Name)
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
)
