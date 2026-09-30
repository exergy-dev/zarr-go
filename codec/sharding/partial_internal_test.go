package sharding

import (
	"bytes"
	"context"
	"math"
	"testing"
)

// TestSlicedFetcherGetRange pins slicedFetcher's range handling: n clamps to
// the blob end (including an off+n that overflows int64), and negative or
// out-of-range arguments are errors rather than slice panics.
func TestSlicedFetcherGetRange(t *testing.T) {
	f := &slicedFetcher{key: "k", base: []byte{0, 1, 2, 3, 4}}
	ok := []struct {
		off, n int64
		want   []byte
	}{
		{0, 5, []byte{0, 1, 2, 3, 4}},
		{1, 2, []byte{1, 2}},
		{3, 100, []byte{3, 4}},
		{1, math.MaxInt64, []byte{1, 2, 3, 4}},
		{5, math.MaxInt64, []byte{}},
		{2, 0, []byte{}},
	}
	for _, tc := range ok {
		got, err := f.GetRange(context.Background(), "k", tc.off, tc.n)
		if err != nil || !bytes.Equal(got, tc.want) {
			t.Errorf("GetRange(%d, %d) = %v, %v; want %v", tc.off, tc.n, got, err, tc.want)
		}
	}
	bad := []struct{ off, n int64 }{{-1, 2}, {0, -1}, {2, -1}, {6, 1}, {math.MaxInt64, math.MaxInt64}}
	for _, tc := range bad {
		if got, err := f.GetRange(context.Background(), "k", tc.off, tc.n); err == nil {
			t.Errorf("GetRange(%d, %d) = %v, want error", tc.off, tc.n, got)
		}
	}
}
