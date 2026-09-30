package metadata

import (
	"encoding/json"
	"testing"

	"github.com/exergy-dev/zarr-go/codec"
)

// The golden tests below pin the exact bytes MarshalArray / MarshalGroup emit
// and the exact results / error strings DecodeArray / DecodeGroup return, so
// a change to the JSON plumbing that alters the on-disk zarr.json format (key
// order, indentation, escaping, raw-value canonicalization) fails loudly
// instead of slipping through the semantic round-trip tests.

func bytesCodecs() []codec.Spec {
	return []codec.Spec{
		{Name: "bytes", Configuration: json.RawMessage(`{ "endian" :  "little" }`)},
		{Name: "gzip", Configuration: json.RawMessage(`{"level":5}`)},
	}
}

func TestMarshalArrayGolden(t *testing.T) {
	cases := []struct {
		name string
		m    *ArrayMeta
		want string
	}{
		{
			name: "nested attributes, html, unicode, extra, dimension names",
			m: &ArrayMeta{
				ZarrFormat: 3,
				NodeType:   "array",
				Shape:      []int64{4, 6},
				DataType:   "float32",
				ChunkGrid: ChunkGrid{Name: "regular", Configuration: ChunkGridRegularCfg{
					ChunkShape: []int64{2, 3},
				}},
				ChunkKeyEnc: ChunkKeyEncoding{Name: "default", Configuration: ChunkKeyEncodingConfig{Separator: "."}},
				FillValue:   json.RawMessage(`"NaN"`),
				Codecs:      bytesCodecs(),
				Attributes: map[string]any{
					"z": 1,
					"list": []any{
						map[string]any{"b": 2, "a": []any{1, map[string]any{"y": true, "x": nil}}},
						"s",
						1.5,
					},
					"html":    `<a href="x">&amp;</a>`,
					"unicode": "héllo 世界 \u2028 🚀",
					"nested":  map[string]any{"k": map[string]any{"j": 1e21, "i": []any{}}},
					"typed":   map[string]int{"b": 1, "a": 2},
					"ints":    []int{3, 1},
					"empty":   map[string]any{},
				},
				Dimensions: []string{"y", "x"},
				Extra: map[string]json.RawMessage{
					"ext_b":       json.RawMessage(" { \"must_understand\" : false ,\n\t \"cfg\": {\"z\": [ 1 , 2 ], \"a\": {} } } "),
					"ext_a":       json.RawMessage(`{"must_understand":false}`),
					"zarr_format": json.RawMessage(`99`), // collides with a known key: dropped
				},
			},
			want: `{
  "attributes": {
    "empty": {},
    "html": "\u003ca href=\"x\"\u003e\u0026amp;\u003c/a\u003e",
    "ints": [
      3,
      1
    ],
    "list": [
      {
        "a": [
          1,
          {
            "x": null,
            "y": true
          }
        ],
        "b": 2
      },
      "s",
      1.5
    ],
    "nested": {
      "k": {
        "i": [],
        "j": 1e+21
      }
    },
    "typed": {
      "a": 2,
      "b": 1
    },
    "unicode": "héllo 世界 \u2028 🚀",
    "z": 1
  },
  "chunk_grid": {
    "name": "regular",
    "configuration": {
      "chunk_shape": [
        2,
        3
      ]
    }
  },
  "chunk_key_encoding": {
    "name": "default",
    "configuration": {
      "separator": "."
    }
  },
  "codecs": [
    {
      "name": "bytes",
      "configuration": {
        "endian": "little"
      }
    },
    {
      "name": "gzip",
      "configuration": {
        "level": 5
      }
    }
  ],
  "data_type": "float32",
  "dimension_names": [
    "y",
    "x"
  ],
  "ext_a": {
    "must_understand": false
  },
  "ext_b": {
    "must_understand": false,
    "cfg": {
      "z": [
        1,
        2
      ],
      "a": {}
    }
  },
  "fill_value": "NaN",
  "node_type": "array",
  "shape": [
    4,
    6
  ],
  "zarr_format": 3
}`,
		},
		{
			name: "rank-0, v2 key encoding, raw fill with spaces, storage_transformers",
			m: &ArrayMeta{
				ZarrFormat:   3,
				NodeType:     "array",
				Shape:        CloneInt64(nil),
				DataType:     "float64",
				ChunkGrid:    ChunkGrid{Name: "regular", Configuration: ChunkGridRegularCfg{ChunkShape: CloneInt64(nil)}},
				ChunkKeyEnc:  ChunkKeyEncoding{Name: "v2"},
				FillValue:    json.RawMessage("  -1.5e3  "),
				Codecs:       []codec.Spec{{Name: "bytes"}},
				StorageTrans: json.RawMessage(" [ ] "),
			},
			want: `{
  "chunk_grid": {
    "name": "regular",
    "configuration": {
      "chunk_shape": []
    }
  },
  "chunk_key_encoding": {
    "name": "v2",
    "configuration": {
      "separator": "."
    }
  },
  "codecs": [
    {
      "name": "bytes"
    }
  ],
  "data_type": "float64",
  "fill_value": -1.5e3,
  "node_type": "array",
  "shape": [],
  "storage_transformers": [],
  "zarr_format": 3
}`,
		},
		{
			name: "defaults filled in",
			m: &ArrayMeta{
				Shape:     []int64{3},
				DataType:  "complex64",
				ChunkGrid: ChunkGrid{Name: "regular", Configuration: ChunkGridRegularCfg{ChunkShape: []int64{3}}},
			},
			want: `{
  "chunk_grid": {
    "name": "regular",
    "configuration": {
      "chunk_shape": [
        3
      ]
    }
  },
  "chunk_key_encoding": {
    "name": "default",
    "configuration": {
      "separator": "/"
    }
  },
  "codecs": null,
  "data_type": "complex64",
  "fill_value": [
    0,
    0
  ],
  "node_type": "array",
  "shape": [
    3
  ],
  "zarr_format": 3
}`,
		},
		{
			name: "raw fill array with spaces",
			m: &ArrayMeta{
				Shape:     []int64{3},
				DataType:  "complex128",
				ChunkGrid: ChunkGrid{Name: "regular", Configuration: ChunkGridRegularCfg{ChunkShape: []int64{3}}},
				FillValue: json.RawMessage("[ 0.5 ,\n -1 ]"),
				Codecs:    []codec.Spec{{Name: "bytes", Configuration: json.RawMessage(`{"endian":"little"}`)}},
			},
			want: `{
  "chunk_grid": {
    "name": "regular",
    "configuration": {
      "chunk_shape": [
        3
      ]
    }
  },
  "chunk_key_encoding": {
    "name": "default",
    "configuration": {
      "separator": "/"
    }
  },
  "codecs": [
    {
      "name": "bytes",
      "configuration": {
        "endian": "little"
      }
    }
  ],
  "data_type": "complex128",
  "fill_value": [
    0.5,
    -1
  ],
  "node_type": "array",
  "shape": [
    3
  ],
  "zarr_format": 3
}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := MarshalArray(tc.m)
			if err != nil {
				t.Fatalf("MarshalArray: %v", err)
			}
			if string(got) != tc.want {
				t.Fatalf("MarshalArray mismatch\ngot:\n%s\nwant:\n%s", got, tc.want)
			}
		})
	}
}

func TestMarshalGroupGolden(t *testing.T) {
	cases := []struct {
		name string
		m    *GroupMeta
		want string
	}{
		{
			name: "empty",
			m:    &GroupMeta{},
			want: `{
  "node_type": "group",
  "zarr_format": 3
}`,
		},
		{
			name: "nested attributes, html, unicode, extra",
			m: &GroupMeta{
				ZarrFormat: 3,
				NodeType:   "group",
				Attributes: map[string]any{
					"b":    []any{map[string]any{"d": 1, "c": []any{"x", map[string]any{"f": false, "e": "<&>"}}}},
					"a":    "ünïcødé ☃",
					"html": "a < b && c > d",
				},
				Extra: map[string]json.RawMessage{
					"ext":        json.RawMessage("{\"must_understand\":false,   \"n\" : [ {\"q\":1} ,{} ] }"),
					"attributes": json.RawMessage(`{"must_understand":false}`), // collides: dropped
				},
			},
			want: `{
  "attributes": {
    "a": "ünïcødé ☃",
    "b": [
      {
        "c": [
          "x",
          {
            "e": "\u003c\u0026\u003e",
            "f": false
          }
        ],
        "d": 1
      }
    ],
    "html": "a \u003c b \u0026\u0026 c \u003e d"
  },
  "ext": {
    "must_understand": false,
    "n": [
      {
        "q": 1
      },
      {}
    ]
  },
  "node_type": "group",
  "zarr_format": 3
}`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := MarshalGroup(tc.m)
			if err != nil {
				t.Fatalf("MarshalGroup: %v", err)
			}
			if string(got) != tc.want {
				t.Fatalf("MarshalGroup mismatch\ngot:\n%s\nwant:\n%s", got, tc.want)
			}
		})
	}
}

// TestDecodeArrayGolden pins what DecodeArray preserves from a document with
// odd whitespace: Extra values, the decoded raw members, and the bytes a
// subsequent MarshalArray produces.
func TestDecodeArrayGolden(t *testing.T) {
	src := []byte("{\n" +
		"  \"zarr_format\" : 3, \"node_type\": \"array\",\n" +
		"  \"shape\": [ 2 ], \"data_type\": \"complex64\",\n" +
		"  \"chunk_grid\": {\"name\": \"regular\", \"configuration\": {\"chunk_shape\": [ 2 ]}},\n" +
		"  \"chunk_key_encoding\": {\"name\": \"v2\"},\n" +
		"  \"fill_value\":   [ 0.5 , -1 ]  ,\n" +
		"  \"codecs\": [ {\"name\": \"bytes\", \"configuration\": { \"endian\" : \"little\" } } ],\n" +
		"  \"attributes\": {\"z\": [ {\"b\": 1, \"a\": \"<&>\"} ], \"a\": \"ü\"},\n" +
		"  \"dimension_names\": [ \"x\" ],\n" +
		"  \"storage_transformers\": [ ],\n" +
		"  \"my_ext\":  { \"must_understand\" : false, \"cfg\" : { \"b\" : [1, 2], \"a\": \"<x>\" } }  \n" +
		"}")
	m, err := DecodeArray(src)
	if err != nil {
		t.Fatalf("DecodeArray: %v", err)
	}
	checks := []struct{ name, got, want string }{
		{"Extra[my_ext]", string(m.Extra["my_ext"]), `{ "must_understand" : false, "cfg" : { "b" : [1, 2], "a": "<x>" } }`},
		{"FillValue", string(m.FillValue), `[0.5,-1]`},
		{"StorageTrans", string(m.StorageTrans), `[]`},
		{"Codecs[0].Configuration", string(m.Codecs[0].Configuration), `{"endian":"little"}`},
		{"ChunkKeyEnc.Separator", m.ChunkKeyEnc.Configuration.Separator, `.`},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %q, want %q", c.name, c.got, c.want)
		}
	}
	if len(m.Extra) != 1 {
		t.Errorf("Extra = %v, want only my_ext", m.Extra)
	}
	out, err := MarshalArray(m)
	if err != nil {
		t.Fatalf("MarshalArray: %v", err)
	}
	const want = `{
  "attributes": {
    "a": "ü",
    "z": [
      {
        "a": "\u003c\u0026\u003e",
        "b": 1
      }
    ]
  },
  "chunk_grid": {
    "name": "regular",
    "configuration": {
      "chunk_shape": [
        2
      ]
    }
  },
  "chunk_key_encoding": {
    "name": "v2",
    "configuration": {
      "separator": "."
    }
  },
  "codecs": [
    {
      "name": "bytes",
      "configuration": {
        "endian": "little"
      }
    }
  ],
  "data_type": "complex64",
  "dimension_names": [
    "x"
  ],
  "fill_value": [
    0.5,
    -1
  ],
  "my_ext": {
    "must_understand": false,
    "cfg": {
      "b": [
        1,
        2
      ],
      "a": "<x>"
    }
  },
  "node_type": "array",
  "shape": [
    2
  ],
  "storage_transformers": [],
  "zarr_format": 3
}`
	if string(out) != want {
		t.Fatalf("round-trip mismatch\ngot:\n%s\nwant:\n%s", out, want)
	}
}

func TestDecodeGroupGolden(t *testing.T) {
	src := []byte(`{"zarr_format":3,"node_type":"group",
		"attributes": { "b": {"y": 1, "x": [ ]}, "a": "<b>" },
		"ext" :  {"must_understand" :false , "k": [ 1 ,2 ] } }`)
	m, err := DecodeGroup(src)
	if err != nil {
		t.Fatalf("DecodeGroup: %v", err)
	}
	if got, want := string(m.Extra["ext"]), `{"must_understand" :false , "k": [ 1 ,2 ] }`; got != want || len(m.Extra) != 1 {
		t.Errorf("Extra = %v (ext %q), want only ext %q", m.Extra, got, want)
	}
	out, err := MarshalGroup(m)
	if err != nil {
		t.Fatalf("MarshalGroup: %v", err)
	}
	const want = `{
  "attributes": {
    "a": "\u003cb\u003e",
    "b": {
      "x": [],
      "y": 1
    }
  },
  "ext": {
    "must_understand": false,
    "k": [
      1,
      2
    ]
  },
  "node_type": "group",
  "zarr_format": 3
}`
	if string(out) != want {
		t.Fatalf("round-trip mismatch\ngot:\n%s\nwant:\n%s", out, want)
	}
}

// TestDecodeErrorsGolden pins the exact error text DecodeArray/DecodeGroup
// return, including the "parse array meta:" / "parse group meta:" prefixes.
func TestDecodeErrorsGolden(t *testing.T) {
	const arr = `"zarr_format": 3, "node_type": "array", "shape": [4], "data_type": "float32",
		"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
		"codecs": [{"name": "bytes"}]`
	const grp = `"zarr_format": 3, "node_type": "group"`
	cases := []struct {
		name   string
		decode func([]byte) error
		src    string
		want   string
	}{
		{"array syntax", decodeArrayErr, `{"zarr_format": 3,`, `parse array meta: unexpected end of JSON input`},
		{"array not object", decodeArrayErr, `[1]`, `parse array meta: json: cannot unmarshal array into Go value of type map[string]json.RawMessage`},
		{"array type error", decodeArrayErr, `{` + arr + `, "fill_value": 0, "dimension_names": "x"}`, `parse array meta: json: cannot unmarshal string into Go struct field ArrayMeta.dimension_names of type []string`},
		{"array ext no must_understand", decodeArrayErr, `{` + arr + `, "fill_value": 0, "ext": {"v": 1}}`, `parse array meta: unknown field "ext": unrecognized extension requires explicit "must_understand": false`},
		{"array ext must_understand true", decodeArrayErr, `{` + arr + `, "fill_value": 0, "ext": {"must_understand": true}}`, `parse array meta: unknown field "ext": unrecognized extension requires explicit "must_understand": false`},
		{"array ext non-object", decodeArrayErr, `{` + arr + `, "fill_value": 0, "ext": "opaque"}`, `parse array meta: unknown field "ext": unrecognized extensions must be an object with "must_understand": false`},
		{"array ext bad must_understand", decodeArrayErr, `{` + arr + `, "fill_value": 0, "ext": {"must_understand": "no"}}`, `parse array meta: unknown field "ext": json: cannot unmarshal string into Go struct field .must_understand of type bool`},
		{"array fill missing", decodeArrayErr, `{` + arr + `}`, `fill_value is required`},
		{"array fill null", decodeArrayErr, `{` + arr + `, "fill_value": null}`, `fill_value is required`},
		{"array wrong node type", decodeArrayErr, `{` + grp + `}`, `not a v3 array document`},
		{"group syntax", decodeGroupErr, `{"zarr_format": 3,`, `parse group meta: unexpected end of JSON input`},
		{"group type error", decodeGroupErr, `{` + grp + `, "attributes": [1]}`, `parse group meta: json: cannot unmarshal array into Go struct field GroupMeta.attributes of type map[string]interface {}`},
		{"group ext no must_understand", decodeGroupErr, `{` + grp + `, "ext": {"v": 1}}`, `parse group meta: unknown field "ext": unrecognized extension requires explicit "must_understand": false`},
		{"group ext non-object", decodeGroupErr, `{` + grp + `, "ext": 7}`, `parse group meta: unknown field "ext": unrecognized extensions must be an object with "must_understand": false`},
		{"group wrong node type", decodeGroupErr, `{"zarr_format": 3, "node_type": "array"}`, `not a v3 group document`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.decode([]byte(tc.src))
			got := "<nil>"
			if err != nil {
				got = err.Error()
			}
			if got != tc.want {
				t.Fatalf("error = %q, want %q", got, tc.want)
			}
		})
	}
}

func decodeArrayErr(raw []byte) error { _, err := DecodeArray(raw); return err }
func decodeGroupErr(raw []byte) error { _, err := DecodeGroup(raw); return err }

// TestMarshalRawEscapingGolden pins how the caller-supplied raw members
// (FillValue, Extra) are emitted: whitespace canonicalized, but '<', '>', '&'
// and U+2028 are not HTML-escaped, unlike Attributes, which go through
// json.Marshal. An invalid raw value is rejected rather than written out as
// invalid JSON.
func TestMarshalRawEscapingGolden(t *testing.T) {
	base := func() *ArrayMeta {
		return &ArrayMeta{
			Shape:     []int64{1},
			DataType:  "float32",
			ChunkGrid: ChunkGrid{Name: "regular", Configuration: ChunkGridRegularCfg{ChunkShape: []int64{1}}},
			Codecs:    []codec.Spec{{Name: "bytes"}},
		}
	}
	html := base()
	html.FillValue = json.RawMessage(`"<&>"`)
	html.Extra = map[string]json.RawMessage{"ext": json.RawMessage("{\"must_understand\":false,\"note\":\"<b> & \u2028\"}")}
	invalid := base()
	invalid.FillValue = json.RawMessage(`nan`)
	cases := []struct {
		name string
		m    *ArrayMeta
		want string
	}{
		{"html in raw members", html, `{
  "chunk_grid": {
    "name": "regular",
    "configuration": {
      "chunk_shape": [
        1
      ]
    }
  },
  "chunk_key_encoding": {
    "name": "default",
    "configuration": {
      "separator": "/"
    }
  },
  "codecs": [
    {
      "name": "bytes"
    }
  ],
  "data_type": "float32",
  "ext": {
    "must_understand": false,
    "note": "<b> & ` + "\u2028" + `"
  },
  "fill_value": "<&>",
  "node_type": "array",
  "shape": [
    1
  ],
  "zarr_format": 3
}`},
		{"invalid raw fill", invalid, `error: json: error calling MarshalJSON for type json.RawMessage: invalid character 'a' in literal null (expecting 'u')`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := MarshalArray(tc.m)
			if err != nil {
				got = []byte("error: " + err.Error())
			}
			if string(got) != tc.want {
				t.Fatalf("MarshalArray mismatch\ngot:\n%s\nwant:\n%s", got, tc.want)
			}
		})
	}
}

// TestDecodeKeyMatchingGolden pins that top-level keys are matched exactly:
// an extension whose name differs from a known field only by case stays in
// Extra without feeding the known field, and a duplicated key keeps only its
// last value.
func TestDecodeKeyMatchingGolden(t *testing.T) {
	const arr = `"zarr_format": 3, "node_type": "array", "shape": [4], "data_type": "float32",
		"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
		"codecs": [{"name": "bytes"}], "fill_value": 0`
	cases := []struct {
		name string
		src  string
		want string
	}{
		{"case-folded extension", `{` + arr + `, "attributes": {"a": 1}, "Attributes": {"must_understand": false}, "SHAPE": {"must_understand": false}}`, `{
  "Attributes": {
    "must_understand": false
  },
  "SHAPE": {
    "must_understand": false
  },
  "attributes": {
    "a": 1
  },
  "chunk_grid": {
    "name": "regular",
    "configuration": {
      "chunk_shape": [
        2
      ]
    }
  },
  "chunk_key_encoding": {
    "name": "default",
    "configuration": {
      "separator": "/"
    }
  },
  "codecs": [
    {
      "name": "bytes"
    }
  ],
  "data_type": "float32",
  "fill_value": 0,
  "node_type": "array",
  "shape": [
    4
  ],
  "zarr_format": 3
}`},
		{"duplicate key", `{` + arr + `, "attributes": {"a": 1}, "attributes": {"b": 2}}`, `{
  "attributes": {
    "b": 2
  },
  "chunk_grid": {
    "name": "regular",
    "configuration": {
      "chunk_shape": [
        2
      ]
    }
  },
  "chunk_key_encoding": {
    "name": "default",
    "configuration": {
      "separator": "/"
    }
  },
  "codecs": [
    {
      "name": "bytes"
    }
  ],
  "data_type": "float32",
  "fill_value": 0,
  "node_type": "array",
  "shape": [
    4
  ],
  "zarr_format": 3
}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got string
			m, err := DecodeArray([]byte(tc.src))
			if err != nil {
				got = "error: " + err.Error()
			} else {
				out, err := MarshalArray(m)
				if err != nil {
					t.Fatalf("MarshalArray: %v", err)
				}
				got = string(out)
			}
			if got != tc.want {
				t.Fatalf("mismatch\ngot:\n%s\nwant:\n%s", got, tc.want)
			}
		})
	}
}
