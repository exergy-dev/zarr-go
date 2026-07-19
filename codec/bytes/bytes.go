// Package bytes implements the Zarr v3 "bytes" array→bytes codec.
//
// The codec serializes an array chunk's element buffer as raw bytes in a
// configured endianness. The package name shadows the stdlib "bytes" package;
// callers that need both should import this one with an alias, e.g.
//
//	import bytescodec "github.com/rkm/zarr-go/codec/bytes"
package bytes

import (
	"encoding/json"
	"fmt"
	"unsafe"

	"github.com/rkm/zarr-go/codec"
)

// hostLittleEndian is true if the host CPU is little-endian. Detected once at
// package init by inspecting the byte layout of a uint16 with value 1.
var hostLittleEndian = func() bool {
	var x uint16 = 1
	return *(*byte)(unsafe.Pointer(&x)) == 1
}()

// Endian represents the configured byte order.
type Endian uint8

const (
	// Little is little-endian byte order.
	Little Endian = iota
	// Big is big-endian byte order.
	Big
)

// Codec is the v3 "bytes" array→bytes codec.
type Codec struct {
	Endian Endian
}

type configJSON struct {
	Endian *string `json:"endian,omitempty"`
}

// Name returns the codec's v3 spec name.
func (c *Codec) Name() string { return "bytes" }

// Configure parses the v3 codec configuration object.
//
// Accepted shape: {"endian":"little"} or {"endian":"big"}. If "endian" is
// absent (or the configuration itself is empty/nil), the default is "little".
func (c *Codec) Configure(raw json.RawMessage) error {
	c.Endian = Little // default
	if len(raw) == 0 {
		return nil
	}
	var cfg configJSON
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return fmt.Errorf("bytes codec: invalid configuration: %w", err)
	}
	if cfg.Endian == nil {
		return nil
	}
	switch *cfg.Endian {
	case "little":
		c.Endian = Little
	case "big":
		c.Endian = Big
	default:
		return fmt.Errorf("bytes codec: invalid endian %q (want \"little\" or \"big\")", *cfg.Endian)
	}
	return nil
}

// swapUnit returns the byte-swap chunk size used for a dtype. Complex numbers
// have two equally-sized real components that are each byte-swapped
// independently, so the swap unit is half the element size.
func swapUnit(dtype string, elemSize int) int {
	switch dtype {
	case "complex64":
		return 4
	case "complex128":
		return 8
	}
	return elemSize
}

// EncodeAB serializes the chunk's element buffer with the configured endianness.
// The returned slice is freshly allocated and owned by the caller.
func (c *Codec) EncodeAB(in codec.ArrayChunk) ([]byte, error) {
	size := codec.DtypeSize(in.Dtype)
	if size == 0 {
		return nil, fmt.Errorf("bytes codec: unsupported dtype %q", in.Dtype)
	}
	if len(in.Data)%size != 0 {
		return nil, fmt.Errorf("bytes codec: data length %d not a multiple of dtype size %d", len(in.Data), size)
	}
	out := make([]byte, len(in.Data))
	if c.needsSwap(size) {
		swapBytes(out, in.Data, swapUnit(in.Dtype, size))
	} else {
		copy(out, in.Data)
	}
	return out, nil
}

// DecodeAB deserializes raw bytes into out.Data, applying the configured
// endianness. out.Data is reused if it already has the expected length;
// otherwise it is reallocated. out.Shape and out.Dtype must be set.
func (c *Codec) DecodeAB(in []byte, out *codec.ArrayChunk) error {
	if out == nil {
		return fmt.Errorf("bytes codec: nil output chunk")
	}
	size := codec.DtypeSize(out.Dtype)
	if size == 0 {
		return fmt.Errorf("bytes codec: unsupported dtype %q", out.Dtype)
	}
	expected := elementCount(out.Shape) * int64(size)
	if int64(len(in)) != expected {
		return fmt.Errorf("bytes codec: input length %d != expected %d for shape %v dtype %s",
			len(in), expected, out.Shape, out.Dtype)
	}
	if int64(cap(out.Data)) >= expected {
		out.Data = out.Data[:expected]
	} else {
		out.Data = make([]byte, expected)
	}
	if c.needsSwap(size) {
		swapBytes(out.Data, in, swapUnit(out.Dtype, size))
	} else {
		copy(out.Data, in)
	}
	return nil
}

// needsSwap reports whether the codec's configured endianness differs from the
// host's, for a multi-byte dtype.
func (c *Codec) needsSwap(elemSize int) bool {
	if elemSize == 1 {
		return false
	}
	wantLittle := c.Endian == Little
	return wantLittle != hostLittleEndian
}

// swapBytes writes src into dst, reversing each unit-sized group of bytes.
// len(src) must equal len(dst) and be a multiple of unit.
//
// Alias safety: dst and src must either be the exact same slice (true
// in-place swap — safe, since each unit-sized group's read and write region
// is identical) or non-overlapping entirely. A partial/offset overlap is
// undefined: the unit==4/8 fast paths (swapBulk32/64) dispatch to SIMD
// routines that process a whole vector-width prefix per call and may read
// ahead of what a byte-at-a-time reversal would have already overwritten,
// corrupting data if dst and src overlap by a non-zero amount less than the
// group size. Every in-tree caller passes a freshly allocated dst distinct
// from src, so this is currently unreachable, not merely untested.
func swapBytes(dst, src []byte, unit int) {
	switch unit {
	case 2:
		for i := 0; i < len(src); i += 2 {
			dst[i] = src[i+1]
			dst[i+1] = src[i]
		}
	case 4:
		done := swapBulk32(dst, src)
		for i := done; i < len(src); i += 4 {
			dst[i] = src[i+3]
			dst[i+1] = src[i+2]
			dst[i+2] = src[i+1]
			dst[i+3] = src[i]
		}
	case 8:
		done := swapBulk64(dst, src)
		for i := done; i < len(src); i += 8 {
			dst[i] = src[i+7]
			dst[i+1] = src[i+6]
			dst[i+2] = src[i+5]
			dst[i+3] = src[i+4]
			dst[i+4] = src[i+3]
			dst[i+5] = src[i+2]
			dst[i+6] = src[i+1]
			dst[i+7] = src[i]
		}
	default:
		// Generic fallback for any other unit size.
		for i := 0; i < len(src); i += unit {
			for j := 0; j < unit; j++ {
				dst[i+j] = src[i+unit-1-j]
			}
		}
	}
}

// elementCount returns product(shape). An empty shape (scalar) yields 1.
func elementCount(shape []int64) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	return n
}

func init() {
	codec.Register("bytes", func() codec.Codec { return &Codec{} })
}
