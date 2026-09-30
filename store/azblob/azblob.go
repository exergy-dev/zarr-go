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
	"strings"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/streaming"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/to"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/blob"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/bloberror"
	"github.com/Azure/azure-sdk-for-go/sdk/storage/azblob/blockblob"

	zstore "github.com/exergy-dev/zarr-go/store"
)

// Config configures a Store.
type Config struct {
	// AccountName is the Azure storage account name, e.g. "mystorageaccount".
	// Required: shared-key auth signs every request with it, even when
	// ServiceURL specifies the endpoint.
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
	if cfg.AccountName == "" {
		return nil, errors.New("azblob: AccountName is required (shared-key auth signs requests with it, even with ServiceURL set)")
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
		prefix:    strings.Trim(cfg.Prefix, "/"),
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
func notFoundErr(key string) error {
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
	k, err := joinKey(s.prefix, key)
	if err != nil {
		return nil, err
	}
	resp, err := s.client.DownloadStream(ctx, s.container, k, nil)
	if err != nil {
		if isNotFound(err) {
			return nil, notFoundErr(key)
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
	if err := checkRange(off, n); err != nil {
		return nil, err
	}
	n = clampLen(off, n)
	k, err := joinKey(s.prefix, key)
	if err != nil {
		return nil, err
	}
	if n == 0 {
		ok, err := s.blobExists(ctx, k)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, notFoundErr(key)
		}
		return []byte{}, nil
	}
	resp, err := s.client.DownloadStream(ctx, s.container, k, &azblob.DownloadStreamOptions{
		Range: blob.HTTPRange{Offset: off, Count: n},
	})
	if err != nil {
		if isNotFound(err) {
			return nil, notFoundErr(key)
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

// GetSuffix implements the store.Store GetSuffix contract. Azure has no suffix
// range ("bytes=-N"), so it reads the blob's size and ETag with
// GetProperties, then downloads the closed range holding the last n bytes,
// conditional on that ETag: a blob replaced in between fails the download
// instead of returning bytes from the wrong offset.
func (s *Store) GetSuffix(ctx context.Context, key string, n int64) ([]byte, error) {
	if err := checkSuffix(n); err != nil {
		return nil, err
	}
	k, err := joinKey(s.prefix, key)
	if err != nil {
		return nil, err
	}
	bbClient := s.client.ServiceClient().NewContainerClient(s.container).NewBlockBlobClient(k)
	props, err := bbClient.GetProperties(ctx, nil)
	if err != nil {
		if isNotFound(err) {
			return nil, notFoundErr(key)
		}
		return nil, fmt.Errorf("azblob: GetProperties %q: %w", k, err)
	}
	var size int64
	if props.ContentLength != nil {
		size = *props.ContentLength
	}
	n = min(n, size)
	if n == 0 {
		// Nothing to read, and a zero Count would mean "to the end".
		return []byte{}, nil
	}
	resp, err := s.client.DownloadStream(ctx, s.container, k, &azblob.DownloadStreamOptions{
		Range: blob.HTTPRange{Offset: size - n, Count: n},
		AccessConditions: &blob.AccessConditions{
			ModifiedAccessConditions: &blob.ModifiedAccessConditions{IfMatch: props.ETag},
		},
	})
	if err != nil {
		if isNotFound(err) {
			return nil, notFoundErr(key)
		}
		return nil, fmt.Errorf("azblob: DownloadStream %q last %d bytes: %w", k, n, err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("azblob: read body %q: %w", k, err)
	}
	return b, nil
}

// Set implements store.Writer.
func (s *Store) Set(ctx context.Context, key string, value []byte) error {
	k, err := joinKey(s.prefix, key)
	if err != nil {
		return err
	}
	_, err = s.client.UploadBuffer(ctx, s.container, k, value, nil)
	if err != nil {
		return fmt.Errorf("azblob: UploadBuffer %q: %w", k, err)
	}
	return nil
}

// SetIfNotExists implements store.Writer. It uses an If-None-Match: *
// access condition; on 412 PreconditionFailed (or BlobAlreadyExists) the
// blob already exists and (false, nil) is returned.
func (s *Store) SetIfNotExists(ctx context.Context, key string, value []byte) (bool, error) {
	k, err := joinKey(s.prefix, key)
	if err != nil {
		return false, err
	}
	bbClient := s.client.ServiceClient().NewContainerClient(s.container).NewBlockBlobClient(k)
	body := streaming.NopCloser(bytes.NewReader(value))
	_, err = bbClient.Upload(ctx, body, &blockblob.UploadOptions{
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

// Delete implements store.Writer. Missing keys are not an error.
func (s *Store) Delete(ctx context.Context, key string) error {
	k, err := joinKey(s.prefix, key)
	if err != nil {
		return err
	}
	_, err = s.client.DeleteBlob(ctx, s.container, k, nil)
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
	k, err := joinKey(s.prefix, key)
	if err != nil {
		return false, err
	}
	return s.blobExists(ctx, k)
}

// List implements store.Store. The supplied prefix is joined with the
// store's configured prefix; yielded keys have the configured prefix
// stripped.
func (s *Store) List(ctx context.Context, prefix string) iter.Seq2[string, error] {
	full := listPrefix(s.prefix, prefix)
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
				k, ok := keyFromObject(s.prefix, *item.Name)
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

// Compile-time interface assertion.
var _ zstore.Writer = (*Store)(nil)
