package zarr

import (
	"errors"
	"fmt"

	"github.com/rkm/zarr-go/codec"
)

// Sentinel errors. Wrap with fmt.Errorf("...: %w", ...) at boundaries.
var (
	// ErrChunkNotFound is returned by stores when a chunk key is missing.
	// The array engine catches this and substitutes the array's fill value.
	ErrChunkNotFound = errors.New("chunk not found")

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

	// ErrCorruptMetadata is returned when JSON parses but doesn't satisfy
	// the spec (e.g., chunk shape with a zero dimension).
	ErrCorruptMetadata = errors.New("corrupt metadata")

	// ErrUnknownCodec is returned at pipeline construction when metadata
	// references a codec name that isn't in the registry.
	ErrUnknownCodec = errors.New("unknown codec")
)

// DecodeError is returned from chunk decode failures with enough context to
// debug a single bad chunk in a 10M-chunk array.
type DecodeError struct {
	Coord []int64
	Codec string
	Err   error
}

func (e *DecodeError) Error() string {
	return fmt.Sprintf("decode chunk %v via codec %q: %v", e.Coord, e.Codec, e.Err)
}

func (e *DecodeError) Unwrap() error { return e.Err }
