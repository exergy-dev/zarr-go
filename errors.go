package zarr

import (
	"errors"
	"fmt"

	"github.com/exergy-dev/zarr-go/codec"
)

// Sentinel errors. Wrap with fmt.Errorf("...: %w", ...) at boundaries.
var (
	// ErrCodecRequiresCGO is raised at array-open time when the metadata
	// references a codec the current build can't handle (e.g., Blosc on
	// the default no-cgo build). It aliases codec.ErrRequiresCGO so callers
	// can errors.Is against either form.
	ErrCodecRequiresCGO = codec.ErrRequiresCGO

	// ErrShapeMismatch is returned when the caller's destination slice is
	// the wrong size for the requested selection.
	ErrShapeMismatch = errors.New("shape mismatch")

	// ErrUnsupportedDtype is returned when metadata declares a dtype that
	// isn't in the Numeric constraint set.
	ErrUnsupportedDtype = errors.New("unsupported dtype")

	// ErrCorruptMetadata is returned when a zarr.json/.zarray/.zgroup
	// document parses as JSON but doesn't satisfy the spec (e.g., a chunk
	// shape with a zero dimension, a missing required field, an
	// unrecognized must_understand extension). Open/OpenArray wrap the
	// underlying internal/metadata error with this sentinel.
	ErrCorruptMetadata = errors.New("corrupt metadata")

	// ErrUnknownCodec is returned at pipeline construction when metadata
	// references a codec name that isn't in the registry. It aliases
	// codec.ErrUnknownCodec so callers can errors.Is against either form.
	ErrUnknownCodec = codec.ErrUnknownCodec

	// ErrReadOnly is returned by Array.Write, CreateArray, and CreateGroup
	// when the underlying store does not implement store.WritableStore.
	ErrReadOnly = errors.New("zarr: store is read-only")
)

// DecodeError is returned from chunk decode failures with enough context to
// debug a single bad chunk in a 10M-chunk array. Err names the codec that
// failed (e.g. `codec "gzip": ...`) when the failure came from one.
type DecodeError struct {
	Coord []int64
	Err   error
}

func (e *DecodeError) Error() string {
	return fmt.Sprintf("decode chunk %v: %v", e.Coord, e.Err)
}

func (e *DecodeError) Unwrap() error { return e.Err }
