// Package transpose implements the Zarr v3 "transpose" array→array codec.
//
// The transpose codec permutes the axes of a chunk according to a configured
// permutation `order`. Encoding reorders axes so that axis i of the output
// corresponds to axis order[i] of the input. Decoding applies the inverse
// permutation.
//
// v3 metadata example:
//
//	{ "name": "transpose", "configuration": { "order": [2, 0, 1] } }
package transpose

import (
	"encoding/json"
	"fmt"

	"github.com/exergy-dev/zarr-go/codec"
)

func init() {
	codec.Register("transpose", func() codec.Codec { return &Codec{} })
}

// Codec is the transpose array→array codec.
type Codec struct {
	order []int
}

// Name returns the v3 codec name.
func (c *Codec) Name() string { return "transpose" }

type config struct {
	Order []int `json:"order"`
}

// Configure parses the codec's configuration JSON. The "order" field must be
// a permutation of [0, rank). Configure validates that order contains each
// index in [0, len(order)) exactly once.
func (c *Codec) Configure(raw json.RawMessage) error {
	var cfg config
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return fmt.Errorf("transpose: parse configuration: %w", err)
		}
	}
	if cfg.Order == nil {
		return fmt.Errorf("transpose: configuration missing required field \"order\"")
	}
	if err := validatePermutation(cfg.Order); err != nil {
		return err
	}
	c.order = append([]int(nil), cfg.Order...)
	return nil
}

// validatePermutation returns an error if order is not a permutation of
// [0, len(order)).
func validatePermutation(order []int) error {
	n := len(order)
	if n == 0 {
		return fmt.Errorf("transpose: \"order\" must be non-empty")
	}
	seen := make([]bool, n)
	for _, v := range order {
		if v < 0 || v >= n {
			return fmt.Errorf("transpose: \"order\" contains out-of-range index %d (rank %d)", v, n)
		}
		if seen[v] {
			return fmt.Errorf("transpose: \"order\" contains duplicate index %d", v)
		}
		seen[v] = true
	}
	return nil
}

// EncodeAA permutes the axes of in according to the configured order and
// writes the result to out. The output shape is
// [in.Shape[order[0]], in.Shape[order[1]], ...].
func (c *Codec) EncodeAA(in codec.ArrayChunk, out *codec.ArrayChunk) error {
	return c.transpose(in, out, c.order)
}

// DecodeAA applies the inverse of the configured permutation to in.
func (c *Codec) DecodeAA(in codec.ArrayChunk, out *codec.ArrayChunk) error {
	inv := invert(c.order)
	return c.transpose(in, out, inv)
}

// invert returns the inverse permutation of p: inv[p[i]] = i.
func invert(p []int) []int {
	inv := make([]int, len(p))
	for i, v := range p {
		inv[v] = i
	}
	return inv
}

// transpose moves bytes from in to out so that out's axis i corresponds to
// in's axis perm[i]. Both shapes must have the same rank as perm and the
// same total number of elements.
func (c *Codec) transpose(in codec.ArrayChunk, out *codec.ArrayChunk, perm []int) error {
	rank := len(perm)
	if len(in.Shape) != rank {
		return fmt.Errorf("transpose: input rank %d does not match permutation rank %d", len(in.Shape), rank)
	}
	elemSize := codec.DtypeSize(in.Dtype)
	if elemSize == 0 {
		return fmt.Errorf("transpose: unsupported dtype %q", in.Dtype)
	}

	// Total element count and input byte length sanity check.
	var nElems int64 = 1
	for _, d := range in.Shape {
		if d < 0 {
			return fmt.Errorf("transpose: negative shape dimension %d", d)
		}
		nElems *= d
	}
	wantBytes := nElems * int64(elemSize)
	if int64(len(in.Data)) != wantBytes {
		return fmt.Errorf("transpose: input data length %d does not match shape %v * %d-byte elements (%d)",
			len(in.Data), in.Shape, elemSize, wantBytes)
	}

	// Compute output shape.
	outShape := make([]int64, rank)
	for i, p := range perm {
		outShape[i] = in.Shape[p]
	}

	// Prepare output buffer.
	if cap(out.Data) >= int(wantBytes) {
		out.Data = out.Data[:wantBytes]
	} else {
		out.Data = make([]byte, wantBytes)
	}
	out.Shape = outShape
	out.Dtype = in.Dtype

	// Zero-element chunk: nothing to copy.
	if nElems == 0 {
		return nil
	}

	// Row-major strides (in elements) for input and output.
	inStrides := rowMajorStrides(in.Shape)
	outStrides := rowMajorStrides(outShape)

	// Walk every logical position in the output via mixed-radix counter
	// over outShape, computing the corresponding input flat index using
	// inStrides indexed by the permutation.
	idx := make([]int64, rank)
	src := in.Data
	dst := out.Data
	es := int64(elemSize)

	// Precompute mapping: at axis i of output, we step inStrides[perm[i]] in input.
	inStepPerOutAxis := make([]int64, rank)
	for i := 0; i < rank; i++ {
		inStepPerOutAxis[i] = inStrides[perm[i]]
	}

	var srcOff int64
	var dstOff int64
	for {
		// Copy one element.
		s := srcOff * es
		d := dstOff * es
		copy(dst[d:d+es], src[s:s+es])

		// Increment mixed-radix index over outShape, last axis fastest.
		axis := rank - 1
		for axis >= 0 {
			idx[axis]++
			srcOff += inStepPerOutAxis[axis]
			dstOff += outStrides[axis]
			if idx[axis] < outShape[axis] {
				break
			}
			// Carry: reset this axis to 0 and propagate.
			srcOff -= inStepPerOutAxis[axis] * outShape[axis]
			dstOff -= outStrides[axis] * outShape[axis]
			idx[axis] = 0
			axis--
		}
		if axis < 0 {
			break
		}
	}
	return nil
}

// rowMajorStrides returns the row-major (C-order) strides in elements for the
// given shape.
func rowMajorStrides(shape []int64) []int64 {
	rank := len(shape)
	strides := make([]int64, rank)
	if rank == 0 {
		return strides
	}
	strides[rank-1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
	return strides
}
