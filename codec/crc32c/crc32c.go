// Package crc32c implements the Zarr v3 "crc32c" bytes→bytes codec.
//
// The codec appends a 4-byte little-endian CRC32C (Castagnoli polynomial)
// trailer to its input on encode, and verifies/strips that trailer on decode.
//
// Spec: https://zarr-specs.readthedocs.io/en/latest/v3/codecs/crc32c/v1.0.html
package crc32c

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"

	"github.com/exergy-dev/zarr-go/codec"
)

// ErrChecksumMismatch is returned by DecodeBB when the stored CRC32C trailer
// does not match the recomputed checksum of the body. It is wrapped in a
// formatted error that also reports the expected and observed digests, so
// callers can use errors.Is to detect corruption while still surfacing the
// detailed message to operators.
var ErrChecksumMismatch = errors.New("crc32c: checksum mismatch")

// castagnoli is the CRC32C polynomial table. Hardware-accelerated on amd64
// and arm64 via the standard library.
var castagnoli = crc32.MakeTable(crc32.Castagnoli)

// Codec implements codec.BytesToBytesCodec for "crc32c". It is stateless:
// the zero value is ready to use and safe for concurrent use.
type Codec struct{}

// Name returns the v3 codec name.
func (*Codec) Name() string { return "crc32c" }

// New builds a crc32c codec from its v3 configuration object. The codec has
// no configurable parameters; an empty configuration (or nil) is accepted,
// and any well-formed JSON object is also accepted and ignored.
func New(raw json.RawMessage) (*Codec, error) {
	if len(raw) > 0 {
		// Accept an empty/arbitrary object; reject only malformed JSON.
		var ignored map[string]json.RawMessage
		if err := json.Unmarshal(raw, &ignored); err != nil {
			return nil, fmt.Errorf("crc32c: invalid configuration: %w", err)
		}
	}
	return &Codec{}, nil
}

// EncodeBB returns a fresh slice consisting of in followed by a 4-byte
// little-endian CRC32C checksum of in.
func (*Codec) EncodeBB(in []byte) ([]byte, error) {
	sum := crc32.Checksum(in, castagnoli)
	out := make([]byte, len(in)+4)
	copy(out, in)
	binary.LittleEndian.PutUint32(out[len(in):], sum)
	return out, nil
}

// DecodeBB verifies the trailing 4-byte little-endian CRC32C checksum and
// returns a fresh copy of the body bytes.
func (*Codec) DecodeBB(in []byte) ([]byte, error) {
	if len(in) < 4 {
		return nil, fmt.Errorf("crc32c: input too short: %d bytes (need at least 4)", len(in))
	}
	split := len(in) - 4
	body := in[:split]
	expected := binary.LittleEndian.Uint32(in[split:])
	got := crc32.Checksum(body, castagnoli)
	if got != expected {
		return nil, fmt.Errorf("%w: expected %x, got %x", ErrChecksumMismatch, expected, got)
	}
	out := make([]byte, len(body))
	copy(out, body)
	return out, nil
}

func init() {
	codec.Register("crc32c", func(raw json.RawMessage) (codec.Codec, error) {
		c, err := New(raw)
		if err != nil {
			return nil, err // not a typed-nil codec.Codec
		}
		return c, nil
	})
}
