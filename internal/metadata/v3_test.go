package metadata

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// TestMarshalArrayDeterministic locks down that two marshals of equivalent
// metadata produce byte-equal output. Regressing this would re-introduce the
// map-iteration-order non-determinism the deterministic encoder is meant to
// fix.
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
// decode→marshal cycle and end up in the rendered JSON.
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
		"future_field": {"version": 99, "knobs": [1, 2, 3]}
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
