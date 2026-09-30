package metadata

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/exergy-dev/zarr-go/codec"
)

// TestMarshalArrayDeterministic checks that two marshals of equivalent
// metadata produce byte-equal output, independent of map iteration order.
func TestMarshalArrayDeterministic(t *testing.T) {
	m := &ArrayMeta{
		ZarrFormat: 3,
		NodeType:   "array",
		Shape:      []int64{4, 4},
		DataType:   "float32",
		ChunkGrid: ChunkGrid{
			Name: "regular",
			Configuration: ChunkGridRegularCfg{
				ChunkShape: []int64{2, 2},
			},
		},
		ChunkKeyEnc: ChunkKeyEncoding{
			Name:          "default",
			Configuration: ChunkKeyEncodingConfig{Separator: "/"},
		},
		FillValue: json.RawMessage("0"),
		Attributes: map[string]any{
			"z": 1, "a": 2, "m": map[string]any{"d": 4, "b": 3},
		},
	}
	first, err := MarshalArray(m)
	if err != nil {
		t.Fatalf("MarshalArray: %v", err)
	}
	for i := 0; i < 16; i++ {
		out, err := MarshalArray(m)
		if err != nil {
			t.Fatalf("MarshalArray (iter %d): %v", i, err)
		}
		if !bytes.Equal(first, out) {
			t.Fatalf("MarshalArray non-deterministic (iter %d):\nfirst:\n%s\nlater:\n%s", i, first, out)
		}
	}
	// And the keys inside attributes must appear in sorted order.
	if !strings.Contains(string(first), `"a": 2,`) ||
		!strings.Contains(string(first), `"m":`) ||
		strings.Index(string(first), `"a":`) > strings.Index(string(first), `"z":`) {
		t.Fatalf("attributes not sorted: %s", first)
	}
}

// TestRoundTripExtraField checks that unknown top-level v3 fields survive a
// decode→marshal cycle and end up in the rendered JSON. Per the v3 spec,
// unrecognized extension fields are only accepted when they carry an
// explicit "must_understand": false.
func TestRoundTripExtraField(t *testing.T) {
	src := []byte(`{
		"zarr_format": 3,
		"node_type": "array",
		"shape": [4],
		"data_type": "float32",
		"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
		"chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
		"fill_value": 0,
		"codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
		"future_field": {"must_understand": false, "version": 99, "knobs": [1, 2, 3]}
	}`)
	m, err := DecodeArray(src)
	if err != nil {
		t.Fatalf("DecodeArray: %v", err)
	}
	if _, ok := m.Extra["future_field"]; !ok {
		t.Fatalf("Extra missing future_field: %v", m.Extra)
	}
	out, err := MarshalArray(m)
	if err != nil {
		t.Fatalf("MarshalArray: %v", err)
	}
	if !bytes.Contains(out, []byte("future_field")) {
		t.Fatalf("future_field dropped on marshal:\n%s", out)
	}
	// And re-decoding the marshaled output should still see the field.
	m2, err := DecodeArray(out)
	if err != nil {
		t.Fatalf("re-DecodeArray: %v", err)
	}
	if _, ok := m2.Extra["future_field"]; !ok {
		t.Fatalf("Extra missing on round-trip: %v", m2.Extra)
	}
}

// TestRejectMustUnderstandExtension checks the negative side of the v3
// extension-field contract: an unrecognized field without an explicit
// "must_understand": false must fail to open, since silently ignoring it
// could mean missing an extension that changes how the data is interpreted.
func TestRejectMustUnderstandExtension(t *testing.T) {
	base := `{
		"zarr_format": 3,
		"node_type": "array",
		"shape": [4],
		"data_type": "float32",
		"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
		"fill_value": 0,
		"codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
	`
	cases := []struct {
		name string
		tail string
	}{
		{"no must_understand key", `, "future_field": {"version": 99}}`},
		{"must_understand true", `, "future_field": {"must_understand": true}}`},
		{"non-object value", `, "future_field": "opaque"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := DecodeArray([]byte(base + tc.tail)); err == nil {
				t.Fatal("expected error for unrecognized extension field, got nil")
			}
		})
	}
}

// v3ArrayDoc builds a minimal valid v3 array document, with chunk_key_encoding
// and fill_value overridable so tests can probe those fields specifically.
func v3ArrayDoc(t *testing.T, chunkKeyEnc, fillValue string) []byte {
	t.Helper()
	tail := ""
	if chunkKeyEnc != "" {
		tail = `, "chunk_key_encoding": ` + chunkKeyEnc
	}
	fv := `0`
	if fillValue != "" {
		fv = fillValue
	}
	return []byte(`{
		"zarr_format": 3,
		"node_type": "array",
		"shape": [4],
		"data_type": "float32",
		"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
		"fill_value": ` + fv + tail + `,
		"codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
	}`)
}

// TestDecodeArrayV2ChunkKeyDefaultSeparator checks that a v3 document
// selecting the "v2" chunk_key_encoding with no explicit separator defaults
// to "." (the v2 convention), not "/" (the "default" encoding's separator).
// Getting this wrong means chunks written by another v3 implementation at
// "arr/0.1" are looked up at "arr/0/1", missed, and silently read back as
// fill — no error.
func TestDecodeArrayV2ChunkKeyDefaultSeparator(t *testing.T) {
	m, err := DecodeArray(v3ArrayDoc(t, `{"name": "v2"}`, ""))
	if err != nil {
		t.Fatalf("DecodeArray: %v", err)
	}
	if m.ChunkKeyEnc.Configuration.Separator != "." {
		t.Fatalf("separator = %q, want \".\"", m.ChunkKeyEnc.Configuration.Separator)
	}
	if got := ChunkKey("arr", m.ChunkKeyEnc, []int64{0, 1}); got != "arr/0.1" {
		t.Fatalf("ChunkKey = %q, want \"arr/0.1\"", got)
	}
}

// TestDecodeArrayRejectsUnknownChunkKeyEncoding checks that an unrecognized
// chunk_key_encoding.name errors instead of silently falling back to the
// "default" layout, which would cause every chunk lookup to miss and the
// array to read back as all-fill with no error.
func TestDecodeArrayRejectsUnknownChunkKeyEncoding(t *testing.T) {
	if _, err := DecodeArray(v3ArrayDoc(t, `{"name": "some_future_encoding"}`, "")); err == nil {
		t.Fatal("expected error for unknown chunk_key_encoding.name")
	}
}

// TestDecodeArrayRequiresFillValue checks that a missing or null fill_value
// is rejected for v3 documents (fill_value is a required member per spec;
// null is not a valid value for any core dtype).
func TestDecodeArrayRequiresFillValue(t *testing.T) {
	t.Run("missing", func(t *testing.T) {
		src := []byte(`{
			"zarr_format": 3,
			"node_type": "array",
			"shape": [4],
			"data_type": "float32",
			"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
			"codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
		}`)
		if _, err := DecodeArray(src); err == nil {
			t.Fatal("expected error for missing fill_value")
		}
	})
	t.Run("null", func(t *testing.T) {
		if _, err := DecodeArray(v3ArrayDoc(t, "", "null")); err == nil {
			t.Fatal("expected error for null fill_value")
		}
	})
}

// TestDecodeArrayRank0 checks a v3 document with an empty ("[]") shape —
// legal per spec as a rank-0 (scalar) array — is accepted, not rejected as
// "shape missing".
func TestDecodeArrayRank0(t *testing.T) {
	src := []byte(`{
		"zarr_format": 3,
		"node_type": "array",
		"shape": [],
		"data_type": "float64",
		"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": []}},
		"fill_value": 0,
		"codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
	}`)
	m, err := DecodeArray(src)
	if err != nil {
		t.Fatalf("DecodeArray rank-0: %v", err)
	}
	if len(m.Shape) != 0 {
		t.Fatalf("Shape = %v, want empty", m.Shape)
	}
	if got := ChunkKey("scalar", m.ChunkKeyEnc, nil); got != "scalar/c" {
		t.Fatalf("ChunkKey = %q, want \"scalar/c\"", got)
	}
}

// TestDecodeArrayShapeMissingVsEmpty checks the distinction DecodeArrayRank0
// relies on: shape absent entirely must still be rejected (only an explicit
// "[]" is a valid rank-0 shape).
func TestDecodeArrayShapeMissingVsEmpty(t *testing.T) {
	src := []byte(`{
		"zarr_format": 3,
		"node_type": "array",
		"data_type": "float64",
		"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": []}},
		"fill_value": 0,
		"codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
	}`)
	if _, err := DecodeArray(src); err == nil {
		t.Fatal("expected error for missing shape")
	}
}

// TestDecodeArrayDimensionNamesLengthMismatch checks dimension_names is
// validated against the array rank when present.
func TestDecodeArrayDimensionNamesLengthMismatch(t *testing.T) {
	src := []byte(`{
		"zarr_format": 3,
		"node_type": "array",
		"shape": [4, 4],
		"data_type": "float32",
		"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2, 2]}},
		"fill_value": 0,
		"codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
		"dimension_names": ["x"]
	}`)
	if _, err := DecodeArray(src); err == nil {
		t.Fatal("expected error for dimension_names length != rank")
	}
}

// TestRejectStorageTransformers confirms the v3 spec invariant that
// storage_transformers MUST be empty/null is enforced. Silent acceptance
// here would let a malformed dataset with future-storage-transformer
// semantics decode without warning.
func TestRejectStorageTransformers(t *testing.T) {
	src := []byte(`{
		"zarr_format": 3,
		"node_type": "array",
		"shape": [4],
		"data_type": "float32",
		"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
		"chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
		"fill_value": 0,
		"codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
		"storage_transformers": [{"name": "made_up"}]
	}`)
	if _, err := DecodeArray(src); err == nil {
		t.Fatalf("expected error for non-empty storage_transformers")
	}
	// Empty array and null are both fine.
	for _, ok := range []string{`[]`, `null`} {
		raw := bytes.Replace(src, []byte(`[{"name": "made_up"}]`), []byte(ok), 1)
		if _, err := DecodeArray(raw); err != nil {
			t.Fatalf("storage_transformers=%s: unexpected error %v", ok, err)
		}
	}
}

// FuzzDecodeArray is a smoke fuzz target — we don't aim for correctness,
// only "no panic / no infinite loop" on arbitrary input.
func FuzzDecodeArray(f *testing.F) {
	seeds := [][]byte{
		[]byte(`{}`),
		[]byte(`{"zarr_format":3,"node_type":"array","shape":[1],"data_type":"int8","chunk_grid":{"name":"regular","configuration":{"chunk_shape":[1]}},"fill_value":0,"codecs":[]}`),
		[]byte(`null`),
		[]byte(`[]`),
		[]byte(``),
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		_, _ = DecodeArray(data)
	})
}

// FuzzDecodeArrayV2 is the v2 counterpart.
func FuzzDecodeArrayV2(f *testing.F) {
	seeds := [][]byte{
		[]byte(`{"zarr_format":2,"shape":[1],"chunks":[1],"dtype":"<f4","compressor":null,"filters":null,"fill_value":0,"order":"C"}`),
		[]byte(`{}`),
		[]byte(``),
	}
	for _, s := range seeds {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		_, _ = DecodeArrayV2(data)
	})
}

// TestChunkKeyPathNormalization checks ChunkKey never emits a leading "/"
// for a root (empty) array path, and tolerates stray slashes around the
// array path.
func TestChunkKeyPathNormalization(t *testing.T) {
	def := ChunkKeyEncoding{Name: "default", Configuration: ChunkKeyEncodingConfig{Separator: "/"}}
	defDot := ChunkKeyEncoding{Name: "default", Configuration: ChunkKeyEncodingConfig{Separator: "."}}
	v2 := ChunkKeyEncoding{Name: "v2", Configuration: ChunkKeyEncodingConfig{Separator: "."}}
	cases := []struct {
		path  string
		enc   ChunkKeyEncoding
		coord []int64
		want  string
	}{
		{"", def, []int64{0, 1}, "c/0/1"},
		{"/", def, []int64{0, 1}, "c/0/1"},
		{"", defDot, []int64{0, 1}, "c.0.1"},
		{"", v2, []int64{0, 1}, "0.1"},
		{"", def, nil, "c"},
		{"", v2, nil, "0"},
		{"a", def, []int64{3}, "a/c/3"},
		{"a/", def, []int64{3}, "a/c/3"},
		{"/a", def, []int64{3}, "a/c/3"},
		{"/a/", v2, []int64{3}, "a/3"},
		{"/g/a/", def, nil, "g/a/c"},
		{"g/a", v2, []int64{1, 2}, "g/a/1.2"},
	}
	for _, tc := range cases {
		if got := ChunkKey(tc.path, tc.enc, tc.coord); got != tc.want {
			t.Errorf("ChunkKey(%q, %s, %v) = %q, want %q", tc.path, tc.enc.Name, tc.coord, got, tc.want)
		}
	}
}

// TestMarshalArrayDefaultFillValue checks MarshalArray fills in the spec
// default fill_value for each dtype family when none is set: false for
// bool, [0,0] for complex, 0 otherwise.
func TestMarshalArrayDefaultFillValue(t *testing.T) {
	for dtype, want := range map[string]string{
		"bool":       "false",
		"complex64":  "[0,0]",
		"complex128": "[0,0]",
		"int32":      "0",
		"float64":    "0",
	} {
		raw, err := MarshalArray(&ArrayMeta{
			Shape:     []int64{4},
			DataType:  dtype,
			ChunkGrid: ChunkGrid{Name: "regular", Configuration: ChunkGridRegularCfg{ChunkShape: []int64{2}}},
			Codecs:    []codec.Spec{{Name: "bytes"}},
		})
		if err != nil {
			t.Fatalf("%s: MarshalArray: %v", dtype, err)
		}
		m, err := DecodeArray(raw)
		if err != nil {
			t.Fatalf("%s: DecodeArray: %v", dtype, err)
		}
		if got := string(m.FillValue); got != want {
			t.Errorf("%s: fill_value = %s, want %s", dtype, got, want)
		}
	}
}
