//go:build cgo && zarrcgo

// This file provides a cgo-backed implementation of the v3 "zstd" codec
// that links against libzstd directly. It REPLACES the pure-Go path in
// zstd.go when the binary is built with the `zarrcgo` build tag and cgo
// is enabled.
//
// Requires libzstd development headers and shared library at build/link
// time. On Debian/Ubuntu:
//
//	sudo apt install libzstd-dev
//
// On macOS with Homebrew:
//
//	brew install zstd
package zstd

/*
#cgo LDFLAGS: -lzstd
#include <stdlib.h>
#include <zstd.h>
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/exergy-dev/zarr-go/internal/declimit"
)

// Codec is the cgo-backed v3 "zstd" bytes→bytes codec. Each call to
// EncodeBB / DecodeBB allocates a Go-side output buffer (sized via
// ZSTD_compressBound / ZSTD_getFrameContentSize) and dispatches the work
// to a per-Codec context allocated lazily on first use.
//
// Both contexts are guarded by a mutex because libzstd's *_CCtx and *_DCtx
// objects are not thread-safe. For high-throughput callers, allocate one
// Codec per goroutine.
type Codec struct {
	// Level is the zstd compression level. 0 means ZSTD_CLEVEL_DEFAULT.
	Level int
	// Checksum, when true, writes the 32-bit content checksum into the
	// frame trailer (ZSTD_c_checksumFlag).
	Checksum bool

	once    sync.Once
	initErr error
	mu      sync.Mutex
	cctx    *C.ZSTD_CCtx
	dctx    *C.ZSTD_DCtx
}

func (c *Codec) init() {
	c.once.Do(func() {
		cctx := C.ZSTD_createCCtx()
		if cctx == nil {
			c.initErr = fmt.Errorf("zstd codec: ZSTD_createCCtx returned NULL")
			return
		}
		dctx := C.ZSTD_createDCtx()
		if dctx == nil {
			C.ZSTD_freeCCtx(cctx)
			c.initErr = fmt.Errorf("zstd codec: ZSTD_createDCtx returned NULL")
			return
		}
		// Apply parameters once at init.
		if c.Level != 0 {
			rc := C.ZSTD_CCtx_setParameter(cctx, C.ZSTD_c_compressionLevel, C.int(c.Level))
			if C.ZSTD_isError(rc) != 0 {
				C.ZSTD_freeCCtx(cctx)
				C.ZSTD_freeDCtx(dctx)
				c.initErr = fmt.Errorf("zstd codec: set compressionLevel: %s", zstdErr(rc))
				return
			}
		}
		flag := C.int(0)
		if c.Checksum {
			flag = 1
		}
		rc := C.ZSTD_CCtx_setParameter(cctx, C.ZSTD_c_checksumFlag, flag)
		if C.ZSTD_isError(rc) != 0 {
			C.ZSTD_freeCCtx(cctx)
			C.ZSTD_freeDCtx(dctx)
			c.initErr = fmt.Errorf("zstd codec: set checksumFlag: %s", zstdErr(rc))
			return
		}
		c.cctx = cctx
		c.dctx = dctx
	})
}

// EncodeBB compresses in and returns a freshly allocated buffer of zstd
// frame bytes. The frame includes the Frame_Content_Size field in its
// header (libzstd populates it from the input size when known via
// ZSTD_compress2 with a single-shot call).
func (c *Codec) EncodeBB(in []byte) ([]byte, error) {
	c.init()
	if c.initErr != nil {
		return nil, c.initErr
	}
	bound := C.ZSTD_compressBound(C.size_t(len(in)))
	out := make([]byte, int(bound))

	c.mu.Lock()
	defer c.mu.Unlock()

	var srcPtr unsafe.Pointer
	if len(in) > 0 {
		srcPtr = unsafe.Pointer(&in[0])
	}
	dstPtr := unsafe.Pointer(&out[0])

	written := C.ZSTD_compress2(
		c.cctx,
		dstPtr, C.size_t(len(out)),
		srcPtr, C.size_t(len(in)),
	)
	if C.ZSTD_isError(written) != 0 {
		return nil, fmt.Errorf("zstd codec: encode: %s", zstdErr(written))
	}
	return out[:int(written)], nil
}

// DecodeBB decompresses a zstd frame in `in` and returns the resulting
// bytes. Requires the encoded frame to carry Frame_Content_Size; that's
// what numcodecs/zstandard writes and what EncodeBB above produces.
func (c *Codec) DecodeBB(in []byte) ([]byte, error) {
	c.init()
	if c.initErr != nil {
		return nil, c.initErr
	}
	if len(in) == 0 {
		return []byte{}, nil
	}
	srcPtr := unsafe.Pointer(&in[0])
	contentSize := C.ZSTD_getFrameContentSize(srcPtr, C.size_t(len(in)))
	// Sentinel values: ZSTD_CONTENTSIZE_UNKNOWN / ZSTD_CONTENTSIZE_ERROR
	// are both very large unsigned values; treat "unknown" by falling back
	// to a streaming-shaped decode loop (rare in our writer's output).
	if contentSize == C.ZSTD_CONTENTSIZE_ERROR {
		return nil, fmt.Errorf("zstd codec: decode: not a valid zstd frame")
	}
	if contentSize == C.ZSTD_CONTENTSIZE_UNKNOWN {
		return c.decodeStreaming(in)
	}
	if err := declimit.CheckDeclaredSize(int64(contentSize)); err != nil {
		return nil, fmt.Errorf("zstd codec: decode: %w", err)
	}

	out := make([]byte, int(contentSize))
	c.mu.Lock()
	defer c.mu.Unlock()

	var dstPtr unsafe.Pointer
	if len(out) > 0 {
		dstPtr = unsafe.Pointer(&out[0])
	}
	written := C.ZSTD_decompressDCtx(
		c.dctx,
		dstPtr, C.size_t(len(out)),
		srcPtr, C.size_t(len(in)),
	)
	if C.ZSTD_isError(written) != 0 {
		return nil, fmt.Errorf("zstd codec: decode: %s", zstdErr(written))
	}
	return out[:int(written)], nil
}

// decodeStreaming handles frames lacking Frame_Content_Size. Allocates
// progressively, doubling output capacity as needed.
func (c *Codec) decodeStreaming(in []byte) ([]byte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	dstream := C.ZSTD_createDStream()
	if dstream == nil {
		return nil, fmt.Errorf("zstd codec: ZSTD_createDStream returned NULL")
	}
	defer C.ZSTD_freeDStream(dstream)

	if rc := C.ZSTD_initDStream(dstream); C.ZSTD_isError(rc) != 0 {
		return nil, fmt.Errorf("zstd codec: ZSTD_initDStream: %s", zstdErr(rc))
	}

	out := make([]byte, 0, len(in)*2)
	chunk := make([]byte, 64*1024)

	inBuf := C.ZSTD_inBuffer{
		src:  unsafe.Pointer(&in[0]),
		size: C.size_t(len(in)),
		pos:  0,
	}

	for inBuf.pos < inBuf.size {
		outBuf := C.ZSTD_outBuffer{
			dst:  unsafe.Pointer(&chunk[0]),
			size: C.size_t(len(chunk)),
			pos:  0,
		}
		rc := C.ZSTD_decompressStream(dstream, &outBuf, &inBuf)
		if C.ZSTD_isError(rc) != 0 {
			return nil, fmt.Errorf("zstd codec: decode: %s", zstdErr(rc))
		}
		out = append(out, chunk[:int(outBuf.pos)]...)
		if maxDec := declimit.Max(); maxDec > 0 && int64(len(out)) > maxDec {
			return nil, fmt.Errorf("zstd codec: decompressed output exceeds %d bytes", maxDec)
		}
		if rc == 0 {
			break
		}
	}
	return out, nil
}

func zstdErr(code C.size_t) string {
	cmsg := C.ZSTD_getErrorName(code)
	return C.GoString(cmsg)
}
