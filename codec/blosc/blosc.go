//go:build cgo && zarrcgo

// Package blosc implements the Zarr v3 "blosc" bytes→bytes codec by
// linking against the C-Blosc library (libblosc).
//
// The Codec is only available when the binary is built with cgo enabled
// AND the `zarrcgo` build tag set. Without those, the stub in
// blosc_nocgo.go takes over and New returns codec.ErrRequiresCGO
// so the array-open path can produce a clear error message.
//
// Build requirements:
//
//	# Debian / Ubuntu
//	sudo apt install libblosc-dev
//
//	# Fedora / RHEL
//	sudo dnf install blosc-devel
//
//	# macOS / Homebrew
//	brew install c-blosc
//
// Then build/test with:
//
//	go build -tags zarrcgo ./...
//	go test  -tags zarrcgo ./codec/blosc/...
package blosc

/*
#cgo LDFLAGS: -lblosc
#include <stdlib.h>
#include <string.h>
#include <blosc.h>
*/
import "C"

import (
	"encoding/json"
	"fmt"
	"sync"
	"unsafe"

	"github.com/exergy-dev/zarr-go/internal/declimit"
)

// Shuffle filter constants matching blosc.h.
const (
	noShuffle  = 0
	shuffle    = 1
	bitShuffle = 2
)

// Codec is the v3 "blosc" bytes→bytes codec. Construct it with New; its
// configuration is fixed at construction and it is safe for concurrent use
// (it calls only libblosc's thread-safe *_ctx entry points).
type Codec struct {
	// cname is the inner compressor: "lz4" | "lz4hc" | "blosclz" |
	// "snappy" | "zlib" | "zstd". Defaults to "lz4".
	cname string
	// clevel is the compression level 0-9. Default 5.
	clevel int
	// shuffleName is "noshuffle" | "shuffle" | "bitshuffle". Default
	// "shuffle".
	shuffleName string
	// typeSize is the element width in bytes. Required to be > 0 when
	// shuffling; passed through to libblosc for any value.
	typeSize int
	// blockSize is the inner-block size hint in bytes. 0 lets libblosc
	// pick automatically.
	blockSize int
}

type configJSON struct {
	CName     *string `json:"cname,omitempty"`
	CLevel    *int    `json:"clevel,omitempty"`
	Shuffle   any     `json:"shuffle,omitempty"`
	TypeSize  *int    `json:"typesize,omitempty"`
	BlockSize *int    `json:"blocksize,omitempty"`
}

// Name returns the codec's v3 spec name.
func (c *Codec) Name() string { return "blosc" }

// New builds a blosc codec from its v3 configuration object and checks
// that libblosc supports the configured compressor.
func New(raw json.RawMessage) (*Codec, error) {
	c := &Codec{
		cname:       "lz4",
		clevel:      5,
		shuffleName: "shuffle",
	}
	if len(raw) > 0 {
		var cfg configJSON
		if err := json.Unmarshal(raw, &cfg); err != nil {
			return nil, fmt.Errorf("blosc codec: invalid configuration: %w", err)
		}
		if cfg.CName != nil {
			c.cname = *cfg.CName
		}
		if cfg.CLevel != nil {
			c.clevel = *cfg.CLevel
		}
		if cfg.Shuffle != nil {
			switch v := cfg.Shuffle.(type) {
			case string:
				c.shuffleName = v
			case float64:
				// Accept the numcodecs/v2 numeric encoding too.
				switch int(v) {
				case noShuffle:
					c.shuffleName = "noshuffle"
				case shuffle:
					c.shuffleName = "shuffle"
				case bitShuffle:
					c.shuffleName = "bitshuffle"
				default:
					return nil, fmt.Errorf("blosc codec: shuffle = %v not in {0,1,2}", v)
				}
			default:
				return nil, fmt.Errorf("blosc codec: shuffle has wrong type %T", cfg.Shuffle)
			}
		}
		if cfg.TypeSize != nil {
			c.typeSize = *cfg.TypeSize
		}
		if cfg.BlockSize != nil {
			c.blockSize = *cfg.BlockSize
		}
	}
	switch c.shuffleName {
	case "noshuffle", "shuffle", "bitshuffle":
	default:
		return nil, fmt.Errorf("blosc codec: unknown shuffle %q", c.shuffleName)
	}
	if c.clevel < 0 || c.clevel > 9 {
		return nil, fmt.Errorf("blosc codec: clevel %d out of range [0,9]", c.clevel)
	}
	if err := checkCompressor(c.cname); err != nil {
		return nil, err
	}
	return c, nil
}

// checkCompressor reports an error if libblosc lacks the named compressor.
func checkCompressor(name string) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	var libVer *C.char
	code := C.blosc_compcode_to_compname(C.blosc_compname_to_compcode(cname), &libVer)
	if code < 0 {
		return fmt.Errorf("blosc codec: compressor %q not available in libblosc", name)
	}
	return nil
}

func (c *Codec) shuffleConst() C.int {
	switch c.shuffleName {
	case "noshuffle":
		return C.int(noShuffle)
	case "bitshuffle":
		return C.int(bitShuffle)
	default:
		return C.int(shuffle)
	}
}

// EncodeBB compresses in via blosc_compress_ctx (the thread-safe variant).
func (c *Codec) EncodeBB(in []byte) ([]byte, error) {
	if len(in) == 0 {
		return []byte{}, nil
	}
	typesize := c.typeSize
	if typesize <= 0 {
		typesize = 1
	}

	dstLen := len(in) + C.BLOSC_MAX_OVERHEAD
	out := getBuf(dstLen)

	cname := C.CString(c.cname)
	defer C.free(unsafe.Pointer(cname))

	written := C.blosc_compress_ctx(
		C.int(c.clevel),
		c.shuffleConst(),
		C.size_t(typesize),
		C.size_t(len(in)),
		unsafe.Pointer(&in[0]),
		unsafe.Pointer(&out[0]),
		C.size_t(dstLen),
		cname,
		C.size_t(c.blockSize),
		C.int(1), // numinternalthreads — caller controls outer parallelism
	)
	if written < 0 {
		putBuf(out)
		return nil, fmt.Errorf("blosc codec: blosc_compress_ctx returned %d", int(written))
	}
	if written == 0 {
		// Output buffer was too small (shouldn't happen given
		// BLOSC_MAX_OVERHEAD), but handle defensively.
		putBuf(out)
		return nil, fmt.Errorf("blosc codec: blosc_compress_ctx output did not fit")
	}
	result := make([]byte, int(written))
	copy(result, out[:int(written)])
	putBuf(out)
	return result, nil
}

// DecodeBB decompresses a blosc frame in `in` via blosc_decompress_ctx.
func (c *Codec) DecodeBB(in []byte) ([]byte, error) {
	if len(in) == 0 {
		return []byte{}, nil
	}
	if len(in) < C.BLOSC_MIN_HEADER_LENGTH {
		return nil, fmt.Errorf("blosc codec: frame too short (%d bytes)", len(in))
	}
	var nbytes, cbytes, blocksize C.size_t
	C.blosc_cbuffer_sizes(unsafe.Pointer(&in[0]), &nbytes, &cbytes, &blocksize)
	if cbytes == 0 || nbytes == 0 {
		// Note: blosc reports 0 for either field on a malformed
		// header, but a legitimately empty payload is also valid.
		// An empty payload was returned above, so treat 0 as error
		// here.
		return nil, fmt.Errorf("blosc codec: invalid header (nbytes=%d, cbytes=%d)", int(nbytes), int(cbytes))
	}
	if int(cbytes) != len(in) {
		return nil, fmt.Errorf("blosc codec: frame length mismatch: header=%d, got=%d", int(cbytes), len(in))
	}
	if err := declimit.CheckDeclaredSize(int64(nbytes)); err != nil {
		return nil, fmt.Errorf("blosc codec: decode: %w", err)
	}

	out := make([]byte, int(nbytes))
	rc := C.blosc_decompress_ctx(
		unsafe.Pointer(&in[0]),
		unsafe.Pointer(&out[0]),
		nbytes,
		C.int(1),
	)
	if rc < 0 {
		return nil, fmt.Errorf("blosc codec: blosc_decompress_ctx returned %d", int(rc))
	}
	return out[:int(rc)], nil
}

// --- output buffer pool -----------------------------------------------------

var bufPool sync.Pool

type pooledBuf struct {
	b []byte
}

func getBuf(n int) []byte {
	v := bufPool.Get()
	if v != nil {
		pb := v.(*pooledBuf)
		if cap(pb.b) >= n {
			return pb.b[:n]
		}
	}
	return make([]byte, n)
}

func putBuf(b []byte) {
	if cap(b) == 0 {
		return
	}
	bufPool.Put(&pooledBuf{b: b[:cap(b)]})
}
