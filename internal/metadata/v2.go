package metadata

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strconv"

	"github.com/rkm/zarr-go/codec"
)

// v2RawArray mirrors the on-disk shape of a Zarr v2 .zarray document — only
// the fields needed for translation to v3.
type v2RawArray struct {
	ZarrFormat         int             `json:"zarr_format"`
	Shape              []int64         `json:"shape"`
	Chunks             []int64         `json:"chunks"`
	Dtype              string          `json:"dtype"`
	Compressor         json.RawMessage `json:"compressor"`
	Filters            json.RawMessage `json:"filters"`
	FillValue          json.RawMessage `json:"fill_value"`
	Order              string          `json:"order"`
	DimensionSeparator string          `json:"dimension_separator"`
}

type v2RawGroup struct {
	ZarrFormat int `json:"zarr_format"`
}

// v2DtypeEntry maps a v2 dtype string to a v3 canonical data_type plus the
// endianness implied by the leading byte-order character.
type v2DtypeEntry struct {
	v3Name string
	endian string // "little", "big", or "" (single-byte types)
}

// v2DtypeTable lists all dtypes the v3 Numeric constraint supports. Only
// entries listed here are accepted; everything else (object, structured,
// strings, datetimes, ...) is rejected.
//
// Byte-order prefix:
//
//	'<' little-endian
//	'>' big-endian
//	'|' not applicable (single-byte; we treat as little for codec config)
//	''  no prefix (rare; treat as native — but we reject to keep things explicit
//	    by requiring an endianness mark for multi-byte types via the table)
var v2DtypeTable = map[string]v2DtypeEntry{
	// signed ints
	"|i1": {"int8", "little"},
	"<i2": {"int16", "little"},
	">i2": {"int16", "big"},
	"<i4": {"int32", "little"},
	">i4": {"int32", "big"},
	"<i8": {"int64", "little"},
	">i8": {"int64", "big"},
	// unsigned ints
	"|u1": {"uint8", "little"},
	"<u2": {"uint16", "little"},
	">u2": {"uint16", "big"},
	"<u4": {"uint32", "little"},
	">u4": {"uint32", "big"},
	"<u8": {"uint64", "little"},
	">u8": {"uint64", "big"},
	// floats
	"<f4": {"float32", "little"},
	">f4": {"float32", "big"},
	"<f8": {"float64", "little"},
	">f8": {"float64", "big"},
	// bool
	"|b1": {"bool", "little"},
}

// v2RawCompressor matches just the discriminator id; the full body is passed
// through as configuration for known codecs.
type v2RawCompressor struct {
	ID string `json:"id"`
}

// DecodeArrayV2 parses a v2 .zarray document and translates it into the
// v3-shaped ArrayMeta. The returned ArrayMeta uses chunk_key_encoding with
// name "v2" and the supplied separator so metadata.ChunkKey() produces correct
// v2 chunk keys.
//
// Returns an error for v2 features outside M2 scope: "F" order, structured
// dtypes, object dtypes, filters, and any compressor other than
// gzip/zstd/zlib/blosc (the latter two are translated but may fail at
// pipeline-build time if not registered).
func DecodeArrayV2(raw []byte) (*ArrayMeta, error) {
	var v v2RawArray
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, fmt.Errorf("parse v2 array meta: %w", err)
	}
	if v.ZarrFormat != 2 {
		return nil, fmt.Errorf("not a v2 array document (zarr_format=%d)", v.ZarrFormat)
	}
	if v.Shape == nil {
		return nil, fmt.Errorf("v2 array shape missing")
	}
	if len(v.Chunks) != len(v.Shape) {
		return nil, fmt.Errorf("v2 chunks rank %d != shape rank %d", len(v.Chunks), len(v.Shape))
	}

	// Order: only "C" supported in M2. Empty defaults to "C" per the v2 spec.
	switch v.Order {
	case "", "C":
		// ok
	case "F":
		return nil, fmt.Errorf("v2 order %q not supported (M2 supports C only)", v.Order)
	default:
		return nil, fmt.Errorf("v2 order %q invalid", v.Order)
	}

	// Filters: must be null or empty array.
	if len(v.Filters) > 0 && !isJSONNull(v.Filters) {
		var arr []json.RawMessage
		if err := json.Unmarshal(v.Filters, &arr); err != nil {
			return nil, fmt.Errorf("v2 filters must be null or array: %w", err)
		}
		if len(arr) > 0 {
			return nil, fmt.Errorf("v2 filters not supported in M2")
		}
	}

	// Dtype translation.
	entry, ok := v2DtypeTable[v.Dtype]
	if !ok {
		return nil, fmt.Errorf("v2 dtype %q not supported", v.Dtype)
	}

	// Compressor → codec pipeline. Always start with the bytes codec.
	endian := entry.endian
	if endian == "" {
		endian = "little"
	}
	bytesSpec, err := makeBytesSpec(endian)
	if err != nil {
		return nil, err
	}
	codecs := []codec.Spec{bytesSpec}

	if len(v.Compressor) > 0 && !isJSONNull(v.Compressor) {
		compSpec, err := translateCompressor(v.Compressor)
		if err != nil {
			return nil, err
		}
		codecs = append(codecs, compSpec)
	}

	// Dimension separator (default ".").
	sep := v.DimensionSeparator
	if sep == "" {
		sep = "."
	}
	if sep != "." && sep != "/" {
		return nil, fmt.Errorf("v2 dimension_separator %q invalid (must be \".\" or \"/\")", sep)
	}

	// Fill value: v2 special encodings ("NaN", "Infinity", "-Infinity")
	// happen to also be the v3 form for floats and pass through unchanged.
	// If absent, default to JSON null (v2's "no fill" convention — unlike
	// v3, DecodeArrayV2 does not require fill_value to be present/non-null).
	// normalizeV2FillValue additionally rewrites the lenient encodings some
	// real-world v2 writers emit (float-formatted ints, 0/1 for bool) into
	// the strict form the shared fill-value parser expects.
	fill := v.FillValue
	if len(fill) == 0 {
		fill = json.RawMessage("null")
	}
	fill = normalizeV2FillValue(fill, entry.v3Name)

	out := &ArrayMeta{
		ZarrFormat: 3,
		NodeType:   "array",
		Shape:      cloneInt64(v.Shape),
		DataType:   entry.v3Name,
		ChunkGrid: ChunkGrid{
			Name: "regular",
			Configuration: ChunkGridRegularCfg{
				ChunkShape: cloneInt64(v.Chunks),
			},
		},
		ChunkKeyEnc: ChunkKeyEncoding{
			Name: "v2",
			Configuration: ChunkKeyEncodingConfig{
				Separator: sep,
			},
		},
		FillValue: fill,
		Codecs:    codecs,
	}
	if err := validateArrayMeta(out); err != nil {
		return nil, err
	}
	return out, nil
}

// normalizeV2FillValue rewrites lenient v2-style fill_value encodings into
// the strict v3 form zarr.parseFillValue expects, matching real-world v2
// writers: some emit integer fill values as float literals (e.g. "0.0" for
// an int32 array) or bool fill values as 0/1 rather than false/true. null
// and anything that doesn't match one of these lenient shapes passes through
// unchanged (float/complex dtypes already use the v3-compatible encoding).
func normalizeV2FillValue(raw json.RawMessage, v3Dtype string) json.RawMessage {
	if isJSONNull(raw) {
		return raw
	}
	switch v3Dtype {
	case "bool":
		var f float64
		if err := json.Unmarshal(raw, &f); err == nil {
			switch f {
			case 0:
				return json.RawMessage("false")
			case 1:
				return json.RawMessage("true")
			}
		}
	case "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64":
		var f float64
		if err := json.Unmarshal(raw, &f); err == nil && f == math.Trunc(f) {
			return json.RawMessage(strconv.FormatInt(int64(f), 10))
		}
	}
	return raw
}

// DecodeGroupV2 parses a .zgroup document and returns a v3-shaped GroupMeta.
func DecodeGroupV2(raw []byte) (*GroupMeta, error) {
	var v v2RawGroup
	if err := json.Unmarshal(raw, &v); err != nil {
		return nil, fmt.Errorf("parse v2 group meta: %w", err)
	}
	if v.ZarrFormat != 2 {
		return nil, fmt.Errorf("not a v2 group document (zarr_format=%d)", v.ZarrFormat)
	}
	return &GroupMeta{
		ZarrFormat: 3,
		NodeType:   "group",
	}, nil
}

// makeBytesSpec builds the canonical bytes-codec Spec for the given endian.
func makeBytesSpec(endian string) (codec.Spec, error) {
	cfg, err := json.Marshal(struct {
		Endian string `json:"endian"`
	}{Endian: endian})
	if err != nil {
		return codec.Spec{}, fmt.Errorf("encode bytes codec config: %w", err)
	}
	return codec.Spec{Name: "bytes", Configuration: cfg}, nil
}

// translateCompressor turns a v2 compressor JSON object into a v3 codec.Spec.
// The full compressor object (minus the "id" discriminator) becomes the
// codec configuration. Recognized ids: gzip, zstd, zlib, blosc.
func translateCompressor(raw json.RawMessage) (codec.Spec, error) {
	var hdr v2RawCompressor
	if err := json.Unmarshal(raw, &hdr); err != nil {
		return codec.Spec{}, fmt.Errorf("parse v2 compressor: %w", err)
	}
	switch hdr.ID {
	case "gzip", "zstd", "zlib", "blosc":
		// ok
	case "":
		return codec.Spec{}, fmt.Errorf("v2 compressor missing \"id\"")
	default:
		return codec.Spec{}, fmt.Errorf("v2 compressor %q not supported", hdr.ID)
	}
	cfg, err := stripCompressorID(raw)
	if err != nil {
		return codec.Spec{}, err
	}
	return codec.Spec{Name: hdr.ID, Configuration: cfg}, nil
}

// stripCompressorID returns the compressor object with the "id" field removed,
// suitable for use as the v3 codec configuration.
func stripCompressorID(raw json.RawMessage) (json.RawMessage, error) {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, fmt.Errorf("parse v2 compressor object: %w", err)
	}
	delete(obj, "id")
	return json.Marshal(obj)
}

// isJSONNull reports whether raw is the literal JSON null token (ignoring
// surrounding whitespace).
func isJSONNull(raw json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}
