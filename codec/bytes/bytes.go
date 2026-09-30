// Package bytes implements the Zarr v3 "bytes" array→bytes codec.
//
// The codec serializes an array chunk's element buffer as raw bytes in a
// configured endianness. The package name shadows the stdlib "bytes" package;
// callers that need both should import this one with an alias, e.g.
//
//	import bytescodec "github.com/exergy-dev/zarr-go/codec/bytes"
package bytes

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math/bits"
	"unsafe"

	"github.com/exergy-dev/zarr-go/codec"
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

// Codec is the v3 "bytes" array→bytes codec. Construct it with New; its
// configuration is fixed at construction and it is safe for concurrent use.
type Codec struct {
	endian Endian
}

type configJSON struct {
	Endian *string `json:"endian,omitempty"`
}

// New builds a bytes codec from its v3 configuration object.
//
// Accepted shape: {"endian":"little"} or {"endian":"big"}. If "endian" is
// absent (or the configuration itself is empty/nil), the default is "little".
func New(raw json.RawMessage) (*Codec, error) {
	c := &Codec{endian: Little}
	if len(raw) == 0 {
		return c, nil
	}
	var cfg configJSON
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("bytes codec: invalid configuration: %w", err)
	}
	if cfg.Endian == nil {
		return c, nil
	}
	switch *cfg.Endian {
	case "little":
		c.endian = Little
	case "big":
		c.endian = Big
	default:
		return nil, fmt.Errorf("bytes codec: invalid endian %q (want \"little\" or \"big\")", *cfg.Endian)
	}
	return c, nil
}

// Name returns the codec's v3 spec name.
func (c *Codec) Name() string { return "bytes" }

// Endian returns the configured byte order.
func (c *Codec) Endian() Endian { return c.endian }

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
	wantLittle := c.endian == Little
	return wantLittle != hostLittleEndian
}

// swapBytes writes src into dst, reversing each unit-sized group of bytes.
// len(src) must equal len(dst) and be a multiple of unit.
//
// For unit 4 and 8, swapBulk32/64 handle a vector-width prefix with SIMD
// where available (amd64 with SSSE3 or AVX2); the rest, or the whole buffer
// elsewhere, goes through the portable word loops swapWords32/64.
//
// Alias safety: dst and src must either be the exact same slice (true
// in-place swap) or not overlap at all. Every path reads a group (or a whole
// vector or word block) before writing it, so in-place is safe. A partial
// overlap is undefined. Every in-tree caller passes a freshly allocated dst
// distinct from src.
func swapBytes(dst, src []byte, unit int) {
	switch unit {
	case 2:
		for i := 0; i < len(src); i += 2 {
			dst[i], dst[i+1] = src[i+1], src[i]
		}
	case 4:
		done := swapBulk32(dst, src)
		swapWords32(dst[done:], src[done:])
	case 8:
		done := swapBulk64(dst, src)
		swapWords64(dst[done:], src[done:])
	default:
		// Generic fallback for any other unit size. Swapping mirrored
		// pairs (j, unit-1-j) keeps it safe in place.
		for i := 0; i < len(src); i += unit {
			for j, k := i, i+unit-1; j <= k; j, k = j+1, k-1 {
				dst[j], dst[k] = src[k], src[j]
			}
		}
	}
}

// swapWords32 reverses each 4-byte group of src into dst using 64-bit loads,
// BSWAP and a 32-bit rotate (which reverses two groups at once), unrolled to
// 32 bytes per iteration. This measured about 5x faster than a byte loop.
// len(src) must be a multiple of 4 and len(dst) >= len(src).
func swapWords32(dst, src []byte) {
	n := len(src) &^ 31
	dst = dst[:len(src)]
	i := 0
	for ; i < n; i += 32 {
		s := src[i : i+32 : i+32]
		d := dst[i : i+32 : i+32]
		v0 := binary.LittleEndian.Uint64(s[0:])
		v1 := binary.LittleEndian.Uint64(s[8:])
		v2 := binary.LittleEndian.Uint64(s[16:])
		v3 := binary.LittleEndian.Uint64(s[24:])
		binary.LittleEndian.PutUint64(d[0:], bits.RotateLeft64(bits.ReverseBytes64(v0), 32))
		binary.LittleEndian.PutUint64(d[8:], bits.RotateLeft64(bits.ReverseBytes64(v1), 32))
		binary.LittleEndian.PutUint64(d[16:], bits.RotateLeft64(bits.ReverseBytes64(v2), 32))
		binary.LittleEndian.PutUint64(d[24:], bits.RotateLeft64(bits.ReverseBytes64(v3), 32))
	}
	for ; i+4 <= len(src); i += 4 {
		binary.LittleEndian.PutUint32(dst[i:], binary.BigEndian.Uint32(src[i:]))
	}
}

// swapWords64 reverses each 8-byte group of src into dst using big-endian
// 64-bit loads (MOVQ+BSWAP on amd64), unrolled to 64 bytes per iteration.
// len(src) must be a multiple of 8 and len(dst) >= len(src).
func swapWords64(dst, src []byte) {
	n := len(src) &^ 63
	dst = dst[:len(src)]
	i := 0
	for ; i < n; i += 64 {
		s := src[i : i+64 : i+64]
		d := dst[i : i+64 : i+64]
		v0 := binary.BigEndian.Uint64(s[0:])
		v1 := binary.BigEndian.Uint64(s[8:])
		v2 := binary.BigEndian.Uint64(s[16:])
		v3 := binary.BigEndian.Uint64(s[24:])
		v4 := binary.BigEndian.Uint64(s[32:])
		v5 := binary.BigEndian.Uint64(s[40:])
		v6 := binary.BigEndian.Uint64(s[48:])
		v7 := binary.BigEndian.Uint64(s[56:])
		binary.LittleEndian.PutUint64(d[0:], v0)
		binary.LittleEndian.PutUint64(d[8:], v1)
		binary.LittleEndian.PutUint64(d[16:], v2)
		binary.LittleEndian.PutUint64(d[24:], v3)
		binary.LittleEndian.PutUint64(d[32:], v4)
		binary.LittleEndian.PutUint64(d[40:], v5)
		binary.LittleEndian.PutUint64(d[48:], v6)
		binary.LittleEndian.PutUint64(d[56:], v7)
	}
	for ; i+8 <= len(src); i += 8 {
		binary.LittleEndian.PutUint64(dst[i:], binary.BigEndian.Uint64(src[i:]))
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
	codec.Register("bytes", func(raw json.RawMessage) (codec.Codec, error) {
		c, err := New(raw)
		if err != nil {
			return nil, err // not a typed-nil codec.Codec
		}
		return c, nil
	})
}
