module github.com/exergy-dev/zarr-go/store/azblob

go 1.25.0

require (
	github.com/Azure/azure-sdk-for-go/sdk/azcore v1.20.0
	github.com/Azure/azure-sdk-for-go/sdk/storage/azblob v1.6.4
	github.com/exergy-dev/zarr-go v0.1.0
)

require (
	github.com/Azure/azure-sdk-for-go/sdk/internal v1.11.2 // indirect
	golang.org/x/net v0.58.0 // indirect
	golang.org/x/text v0.41.0 // indirect
)

// Development: build against the zarr-go module in this repository, so a
// change to the store interface and to this backend land in one commit.
//
// Consumers never see a replace directive (it applies only when this is the
// main module), so a release cannot carry the placeholder require above.
