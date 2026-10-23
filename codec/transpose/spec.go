package transpose

import (
	"encoding/json"

	"github.com/exergy-dev/zarr-go/codec"
)

// Spec returns the metadata entry for a transpose codec with the given axis
// order, for zarr.WithCodecs. order must be a permutation of [0, rank); an
// invalid order is reported when the array is created.
func Spec(order ...int) codec.Spec {
	if order == nil {
		order = []int{}
	}
	raw, _ := json.Marshal(struct {
		Order []int `json:"order"`
	}{order}) // a slice of ints cannot fail to marshal
	return codec.Spec{Name: "transpose", Configuration: raw}
}
