package metadata

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestDecodeArrayV2_MinimalNoCompressor(t *testing.T) {
	raw := []byte(`{
		"zarr_format": 2,
		"shape": [10, 20],
		"chunks": [5, 10],
		"dtype": "<f4",
		"compressor": null,
		"filters": null,
		"fill_value": 0,
		"order": "C"
	}`)
	m, err := DecodeArrayV2(raw)
	if err != nil {
		t.Fatalf("DecodeArrayV2: %v", err)
	}
	if got, want := m.ZarrFormat, 3; got != want {
		t.Errorf("ZarrFormat = %d, want %d", got, want)
	}
	if got, want := m.NodeType, "array"; got != want {
		t.Errorf("NodeType = %q, want %q", got, want)
	}
	if got, want := m.DataType, "float32"; got != want {
		t.Errorf("DataType = %q, want %q", got, want)
	}
	if got, want := len(m.Shape), 2; got != want {
		t.Fatalf("len(Shape) = %d, want %d", got, want)
	}
	if m.Shape[0] != 10 || m.Shape[1] != 20 {
		t.Errorf("Shape = %v, want [10 20]", m.Shape)
	}
	if got, want := m.ChunkGrid.Name, "regular"; got != want {
		t.Errorf("ChunkGrid.Name = %q, want %q", got, want)
	}
	cs := m.ChunkGrid.Configuration.ChunkShape
	if len(cs) != 2 || cs[0] != 5 || cs[1] != 10 {
		t.Errorf("ChunkShape = %v, want [5 10]", cs)
	}
	if got, want := m.ChunkKeyEnc.Name, "v2"; got != want {
		t.Errorf("ChunkKeyEnc.Name = %q, want %q", got, want)
	}
	if got, want := m.ChunkKeyEnc.Configuration.Separator, "."; got != want {
		t.Errorf("ChunkKeyEnc.Configuration.Separator = %q, want %q", got, want)
	}
	if got, want := len(m.Codecs), 1; got != want {
		t.Fatalf("len(Codecs) = %d, want %d (bytes only)", got, want)
	}
	if got, want := m.Codecs[0].Name, "bytes"; got != want {
		t.Errorf("Codecs[0].Name = %q, want %q", got, want)
	}
	var bcfg struct {
		Endian string `json:"endian"`
	}
	if err := json.Unmarshal(m.Codecs[0].Configuration, &bcfg); err != nil {
		t.Fatalf("unmarshal bytes config: %v", err)
	}
	if got, want := bcfg.Endian, "little"; got != want {
		t.Errorf("bytes endian = %q, want %q", got, want)
	}
}

func TestDecodeArrayV2_GzipCompressor(t *testing.T) {
	raw := []byte(`{
		"zarr_format": 2,
		"shape": [4],
		"chunks": [2],
		"dtype": "<i4",
		"compressor": {"id": "gzip", "level": 5},
		"filters": null,
		"fill_value": 0,
		"order": "C"
	}`)
	m, err := DecodeArrayV2(raw)
	if err != nil {
		t.Fatalf("DecodeArrayV2: %v", err)
	}
	if got, want := len(m.Codecs), 2; got != want {
		t.Fatalf("len(Codecs) = %d, want %d", got, want)
	}
	if m.Codecs[0].Name != "bytes" {
		t.Errorf("Codecs[0].Name = %q, want %q", m.Codecs[0].Name, "bytes")
	}
	if m.Codecs[1].Name != "gzip" {
		t.Errorf("Codecs[1].Name = %q, want %q", m.Codecs[1].Name, "gzip")
	}
	var gz struct {
		Level int `json:"level"`
		ID    any `json:"id"`
	}
	if err := json.Unmarshal(m.Codecs[1].Configuration, &gz); err != nil {
		t.Fatalf("unmarshal gzip config: %v", err)
	}
	if gz.Level != 5 {
		t.Errorf("gzip level = %d, want 5", gz.Level)
	}
	if gz.ID != nil {
		t.Errorf("gzip config still contains id: %v", gz.ID)
	}
}

func TestDecodeArrayV2_BigEndianInt32(t *testing.T) {
	raw := []byte(`{
		"zarr_format": 2,
		"shape": [3],
		"chunks": [3],
		"dtype": ">i4",
		"compressor": null,
		"filters": null,
		"fill_value": 0,
		"order": "C"
	}`)
	m, err := DecodeArrayV2(raw)
	if err != nil {
		t.Fatalf("DecodeArrayV2: %v", err)
	}
	if got, want := m.DataType, "int32"; got != want {
		t.Errorf("DataType = %q, want %q", got, want)
	}
	if len(m.Codecs) != 1 || m.Codecs[0].Name != "bytes" {
		t.Fatalf("Codecs = %+v, want [bytes]", m.Codecs)
	}
	var bcfg struct {
		Endian string `json:"endian"`
	}
	if err := json.Unmarshal(m.Codecs[0].Configuration, &bcfg); err != nil {
		t.Fatalf("unmarshal bytes config: %v", err)
	}
	if got, want := bcfg.Endian, "big"; got != want {
		t.Errorf("bytes endian = %q, want %q", got, want)
	}
}

func TestDecodeArrayV2_RejectFortranOrder(t *testing.T) {
	raw := []byte(`{
		"zarr_format": 2,
		"shape": [4],
		"chunks": [2],
		"dtype": "<f4",
		"compressor": null,
		"filters": null,
		"fill_value": 0,
		"order": "F"
	}`)
	_, err := DecodeArrayV2(raw)
	if err == nil {
		t.Fatalf("expected error for order=F, got nil")
	}
	if !strings.Contains(err.Error(), "order") {
		t.Errorf("error = %q, want it to mention order", err)
	}
}

func TestDecodeArrayV2_RejectUnknownDtype(t *testing.T) {
	raw := []byte(`{
		"zarr_format": 2,
		"shape": [4],
		"chunks": [2],
		"dtype": "O",
		"compressor": null,
		"filters": null,
		"fill_value": null,
		"order": "C"
	}`)
	_, err := DecodeArrayV2(raw)
	if err == nil {
		t.Fatalf("expected error for dtype=O, got nil")
	}
	if !strings.Contains(err.Error(), "dtype") {
		t.Errorf("error = %q, want it to mention dtype", err)
	}
}

func TestDecodeArrayV2_RejectFilters(t *testing.T) {
	raw := []byte(`{
		"zarr_format": 2,
		"shape": [4],
		"chunks": [2],
		"dtype": "<f4",
		"compressor": null,
		"filters": [{"id": "delta"}],
		"fill_value": 0,
		"order": "C"
	}`)
	_, err := DecodeArrayV2(raw)
	if err == nil {
		t.Fatalf("expected error for non-empty filters, got nil")
	}
	if !strings.Contains(err.Error(), "filters") {
		t.Errorf("error = %q, want it to mention filters", err)
	}
}

func TestDecodeArrayV2_RejectWrongFormat(t *testing.T) {
	raw := []byte(`{"zarr_format": 3, "shape": [1], "chunks": [1], "dtype": "<f4"}`)
	_, err := DecodeArrayV2(raw)
	if err == nil {
		t.Fatalf("expected error for zarr_format=3")
	}
}

func TestDecodeArrayV2_DefaultDimensionSeparator(t *testing.T) {
	raw := []byte(`{
		"zarr_format": 2,
		"shape": [4],
		"chunks": [2],
		"dtype": "<f4",
		"compressor": null,
		"filters": null,
		"fill_value": 0,
		"order": "C"
	}`)
	m, err := DecodeArrayV2(raw)
	if err != nil {
		t.Fatalf("DecodeArrayV2: %v", err)
	}
	if got, want := m.ChunkKeyEnc.Configuration.Separator, "."; got != want {
		t.Errorf("default separator = %q, want %q", got, want)
	}
}

func TestDecodeArrayV2_SlashDimensionSeparator(t *testing.T) {
	raw := []byte(`{
		"zarr_format": 2,
		"shape": [4],
		"chunks": [2],
		"dtype": "<f4",
		"compressor": null,
		"filters": null,
		"fill_value": 0,
		"order": "C",
		"dimension_separator": "/"
	}`)
	m, err := DecodeArrayV2(raw)
	if err != nil {
		t.Fatalf("DecodeArrayV2: %v", err)
	}
	if got, want := m.ChunkKeyEnc.Configuration.Separator, "/"; got != want {
		t.Errorf("separator = %q, want %q", got, want)
	}
}

func TestDecodeArrayV2_ZstdAndZlibPassthrough(t *testing.T) {
	for _, id := range []string{"zstd", "zlib", "blosc"} {
		raw := []byte(`{
			"zarr_format": 2,
			"shape": [4],
			"chunks": [2],
			"dtype": "<f4",
			"compressor": {"id": "` + id + `", "level": 3},
			"filters": null,
			"fill_value": 0,
			"order": "C"
		}`)
		m, err := DecodeArrayV2(raw)
		if err != nil {
			t.Fatalf("%s: DecodeArrayV2: %v", id, err)
		}
		if len(m.Codecs) != 2 || m.Codecs[1].Name != id {
			t.Errorf("%s: codecs = %+v, want bytes + %s", id, m.Codecs, id)
		}
	}
}

func TestDecodeArrayV2_FillValuePassthrough(t *testing.T) {
	cases := []struct {
		name string
		fv   string
	}{
		{"int", `42`},
		{"null", `null`},
		{"NaN", `"NaN"`},
		{"float", `3.14`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte(`{
				"zarr_format": 2,
				"shape": [1],
				"chunks": [1],
				"dtype": "<f4",
				"compressor": null,
				"filters": null,
				"fill_value": ` + tc.fv + `,
				"order": "C"
			}`)
			m, err := DecodeArrayV2(raw)
			if err != nil {
				t.Fatalf("DecodeArrayV2: %v", err)
			}
			if string(m.FillValue) != tc.fv {
				t.Errorf("FillValue = %q, want %q", string(m.FillValue), tc.fv)
			}
		})
	}
}

// TestDecodeArrayV2_FillValueLeniency checks normalizeV2FillValue rewrites
// the lenient encodings real-world v2 writers emit — float-formatted ints,
// 0/1 for bool — into the strict form the shared v3 fill-value parser
// expects, while leaving float/complex/null encodings untouched.
func TestDecodeArrayV2_FillValueLeniency(t *testing.T) {
	cases := []struct {
		name  string
		dtype string
		fv    string
		want  string
	}{
		{"int_float_formatted", "<i4", `0.0`, `0`},
		{"int_float_formatted_nonzero", "<i4", `5.0`, `5`},
		{"bool_zero", "|b1", `0`, `false`},
		{"bool_one", "|b1", `1`, `true`},
		{"bool_already_strict", "|b1", `true`, `true`},
		{"int_already_strict", "<i4", `7`, `7`},
		{"float_untouched", "<f4", `3.14`, `3.14`},
		{"null_untouched", "<i4", `null`, `null`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte(`{
				"zarr_format": 2,
				"shape": [1],
				"chunks": [1],
				"dtype": "` + tc.dtype + `",
				"compressor": null,
				"filters": null,
				"fill_value": ` + tc.fv + `,
				"order": "C"
			}`)
			m, err := DecodeArrayV2(raw)
			if err != nil {
				t.Fatalf("DecodeArrayV2: %v", err)
			}
			if string(m.FillValue) != tc.want {
				t.Errorf("FillValue = %q, want %q", string(m.FillValue), tc.want)
			}
		})
	}
}

// TestDecodeArrayV2_RejectNegativeShape checks that v2's shared validation
// (routed through validateArrayMeta, the same validator DecodeArray uses)
// catches a negative shape dimension — a gap that existed when v2 had its
// own, less thorough, inline checks.
func TestDecodeArrayV2_RejectNegativeShape(t *testing.T) {
	raw := []byte(`{
		"zarr_format": 2,
		"shape": [-4],
		"chunks": [1],
		"dtype": "<f4",
		"compressor": null,
		"filters": null,
		"fill_value": 0,
		"order": "C"
	}`)
	if _, err := DecodeArrayV2(raw); err == nil {
		t.Fatal("expected error for negative shape")
	}
}

// TestDecodeArrayV2_Rank0 checks a v2 document with an empty ("[]") shape —
// zarr-python's convention for a scalar array — is accepted.
func TestDecodeArrayV2_Rank0(t *testing.T) {
	raw := []byte(`{
		"zarr_format": 2,
		"shape": [],
		"chunks": [],
		"dtype": "<f8",
		"compressor": null,
		"filters": null,
		"fill_value": 0,
		"order": "C"
	}`)
	m, err := DecodeArrayV2(raw)
	if err != nil {
		t.Fatalf("DecodeArrayV2 rank-0: %v", err)
	}
	if len(m.Shape) != 0 {
		t.Fatalf("Shape = %v, want empty", m.Shape)
	}
	if got := ChunkKey("scalar", m.ChunkKeyEnc, nil); got != "scalar/0" {
		t.Fatalf("ChunkKey = %q, want \"scalar/0\"", got)
	}
}

func TestDecodeGroupV2(t *testing.T) {
	raw := []byte(`{"zarr_format": 2}`)
	g, err := DecodeGroupV2(raw)
	if err != nil {
		t.Fatalf("DecodeGroupV2: %v", err)
	}
	if g.ZarrFormat != 3 {
		t.Errorf("ZarrFormat = %d, want 3 (translated)", g.ZarrFormat)
	}
	if g.NodeType != "group" {
		t.Errorf("NodeType = %q, want %q", g.NodeType, "group")
	}
}

func TestDecodeGroupV2_RejectWrongFormat(t *testing.T) {
	if _, err := DecodeGroupV2([]byte(`{"zarr_format": 3}`)); err == nil {
		t.Fatalf("expected error for zarr_format=3")
	}
}

func TestChunkKey_V2Encoding(t *testing.T) {
	enc := ChunkKeyEncoding{
		Name:          "v2",
		Configuration: ChunkKeyEncodingConfig{Separator: "."},
	}
	got := ChunkKey("arr", enc, []int64{1, 2, 3})
	if want := "arr/1.2.3"; got != want {
		t.Errorf("ChunkKey = %q, want %q", got, want)
	}
}

func TestChunkKey_V2Encoding_Slash(t *testing.T) {
	enc := ChunkKeyEncoding{
		Name:          "v2",
		Configuration: ChunkKeyEncodingConfig{Separator: "/"},
	}
	got := ChunkKey("arr", enc, []int64{1, 2, 3})
	if want := "arr/1/2/3"; got != want {
		t.Errorf("ChunkKey = %q, want %q", got, want)
	}
}
