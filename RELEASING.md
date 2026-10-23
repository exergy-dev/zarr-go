# Releasing

The repository holds four modules: the root `github.com/exergy-dev/zarr-go`
and the nested `store/s3`, `store/gcs` and `store/azblob`. Each nested
module requires the root module; during development its `go.mod` points
that requirement at the working tree with
`replace github.com/exergy-dev/zarr-go => ../..` and a placeholder version.
Replace directives are ignored when a module is a dependency, so a nested
module must require a real, tagged root version before it is tagged
itself. Release in this order:

1. Tag the root module `vX.Y.Z` and push the tag.
2. In each of `store/s3`, `store/gcs` and `store/azblob`: set the
   `require github.com/exergy-dev/zarr-go` line to `vX.Y.Z`, delete the
   `replace` directive, and run `go mod tidy` and `go test ./...`. The
   tests now build against the tagged root, so a backend that depends on
   unreleased root changes fails here rather than for users.
3. Commit those `go.mod`/`go.sum` changes and tag each nested module with
   its directory as the tag prefix: `store/s3/vX.Y.Z`, `store/gcs/vX.Y.Z`,
   `store/azblob/vX.Y.Z`. Push the tags.
4. Restore the `replace` directives on the development branch (keeping the
   new `require` versions) so the nested modules build against the working
   tree again.

The CHANGELOG's `[Unreleased]` heading becomes `[X.Y.Z] - YYYY-MM-DD` in the
same commit as step 1.

A nested module can be released on its own, for a backend-only fix, by
repeating steps 2-4 against the latest root tag. It needs a new root tag
only when it depends on root changes that are not yet released.
