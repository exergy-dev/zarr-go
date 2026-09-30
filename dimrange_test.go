package zarr_test

import (
	"testing"

	zarr "github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/codec"
)

// TestDimRangeIsCodecDimRange pins that the selection range type and the
// codec's partial-decode range type are one type, so no conversion is ever
// needed between them.
func TestDimRangeIsCodecDimRange(t *testing.T) {
	length := func(r codec.DimRange) int64 { return r.Length() }
	if got := length(zarr.DimRange{Start: 1, Stop: 10, Step: 3}); got != 3 {
		t.Fatalf("Length() = %d, want 3", got)
	}
}
