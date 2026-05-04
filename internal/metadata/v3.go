// Package metadata parses Zarr v3 array and group JSON.
//
// The exposed types are intentionally narrow — only the fields the engine
// needs. Unknown fields are preserved in Extra for round-tripping but are
// not interpreted.
package metadata

import (
	"encoding/json"
	"fmt"

	"github.com/rkm/zarr-go/codec"
)

// NodeType reports whether a zarr.json describes a group or an array.
type NodeType int

const (
	NodeUnknown NodeType = iota
	NodeGroup
	NodeArray
)

// Header is the small subset shared by group and array zarr.json documents.
type Header struct {
	ZarrFormat int    `json:"zarr_format"`
	NodeType   string `json:"node_type"`
}

// ArrayMeta is the v3 zarr.json for an array.
type ArrayMeta struct {
	ZarrFormat     int               `json:"zarr_format"`
	NodeType       string            `json:"node_type"`
	Shape          []int64           `json:"shape"`
	DataType       string            `json:"data_type"`
	ChunkGrid      ChunkGrid         `json:"chunk_grid"`
	ChunkKeyEnc    ChunkKeyEncoding  `json:"chunk_key_encoding"`
	FillValue      json.RawMessage   `json:"fill_value"`
	Codecs         []codec.Spec      `json:"codecs"`
	Attributes     map[string]any    `json:"attributes,omitempty"`
	StorageTrans   json.RawMessage   `json:"storage_transformers,omitempty"`
	Dimensions     []string          `json:"dimension_names,omitempty"`
}

// GroupMeta is the v3 zarr.json for a group.
type GroupMeta struct {
	ZarrFormat int            `json:"zarr_format"`
	NodeType   string         `json:"node_type"`
	Attributes map[string]any `json:"attributes,omitempty"`
}

// ChunkGrid describes how the array is partitioned into chunks. Only the
// "regular" grid is supported in M1.
type ChunkGrid struct {
	Name          string             `json:"name"`
	Configuration ChunkGridRegularCfg `json:"configuration"`
}

type ChunkGridRegularCfg struct {
	ChunkShape []int64 `json:"chunk_shape"`
}

// ChunkKeyEncoding controls how chunk coordinates map to keys in the store.
type ChunkKeyEncoding struct {
	Name          string                  `json:"name"`
	Configuration ChunkKeyEncodingConfig  `json:"configuration"`
}

type ChunkKeyEncodingConfig struct {
	Separator string `json:"separator"`
}

// PeekNodeType inspects the top-level "node_type" field.
func PeekNodeType(raw []byte) (NodeType, error) {
	var h Header
	if err := json.Unmarshal(raw, &h); err != nil {
		return NodeUnknown, fmt.Errorf("parse header: %w", err)
	}
	if h.ZarrFormat != 3 {
		return NodeUnknown, fmt.Errorf("zarr_format %d not supported (M1 reads v3 only)", h.ZarrFormat)
	}
	switch h.NodeType {
	case "array":
		return NodeArray, nil
	case "group":
		return NodeGroup, nil
	}
	return NodeUnknown, fmt.Errorf("unknown node_type %q", h.NodeType)
}

// DecodeArray parses a zarr.json array document and validates required fields.
func DecodeArray(raw []byte) (*ArrayMeta, error) {
	m := &ArrayMeta{}
	if err := json.Unmarshal(raw, m); err != nil {
		return nil, fmt.Errorf("parse array meta: %w", err)
	}
	if m.ZarrFormat != 3 || m.NodeType != "array" {
		return nil, fmt.Errorf("not a v3 array document")
	}
	if len(m.Shape) == 0 {
		return nil, fmt.Errorf("array shape missing")
	}
	if m.ChunkGrid.Name != "regular" {
		return nil, fmt.Errorf("unsupported chunk_grid %q (M1 supports regular only)", m.ChunkGrid.Name)
	}
	if len(m.ChunkGrid.Configuration.ChunkShape) != len(m.Shape) {
		return nil, fmt.Errorf("chunk_shape rank %d != array rank %d",
			len(m.ChunkGrid.Configuration.ChunkShape), len(m.Shape))
	}
	for i, n := range m.ChunkGrid.Configuration.ChunkShape {
		if n <= 0 {
			return nil, fmt.Errorf("chunk_shape[%d] must be positive, got %d", i, n)
		}
	}
	if m.ChunkKeyEnc.Name == "" {
		m.ChunkKeyEnc.Name = "default"
		m.ChunkKeyEnc.Configuration.Separator = "/"
	}
	if m.ChunkKeyEnc.Configuration.Separator == "" {
		m.ChunkKeyEnc.Configuration.Separator = "/"
	}
	return m, nil
}

// MarshalArray serializes an ArrayMeta into the canonical v3 zarr.json form.
//
// Required fields (zarr_format, node_type, shape, data_type, chunk_grid,
// chunk_key_encoding, fill_value, codecs) are always emitted. Attributes,
// storage_transformers and dimension_names are omitted when empty.
func MarshalArray(m *ArrayMeta) ([]byte, error) {
	out := *m
	if out.ZarrFormat == 0 {
		out.ZarrFormat = 3
	}
	if out.NodeType == "" {
		out.NodeType = "array"
	}
	if out.ChunkKeyEnc.Name == "" {
		out.ChunkKeyEnc.Name = "default"
	}
	if out.ChunkKeyEnc.Configuration.Separator == "" {
		out.ChunkKeyEnc.Configuration.Separator = "/"
	}
	if len(out.FillValue) == 0 {
		out.FillValue = json.RawMessage("0")
	}
	return json.MarshalIndent(out, "", "  ")
}

// MarshalGroup serializes a GroupMeta into the canonical v3 zarr.json form.
func MarshalGroup(m *GroupMeta) ([]byte, error) {
	out := *m
	if out.ZarrFormat == 0 {
		out.ZarrFormat = 3
	}
	if out.NodeType == "" {
		out.NodeType = "group"
	}
	return json.MarshalIndent(out, "", "  ")
}

// DecodeGroup parses a zarr.json group document.
func DecodeGroup(raw []byte) (*GroupMeta, error) {
	m := &GroupMeta{}
	if err := json.Unmarshal(raw, m); err != nil {
		return nil, fmt.Errorf("parse group meta: %w", err)
	}
	if m.ZarrFormat != 3 || m.NodeType != "group" {
		return nil, fmt.Errorf("not a v3 group document")
	}
	return m, nil
}

// ChunkKey returns the store key for the chunk at coord under arrayPath,
// honoring the array's chunk_key_encoding.
func ChunkKey(arrayPath string, enc ChunkKeyEncoding, coord []int64) string {
	sep := enc.Configuration.Separator
	if sep == "" {
		sep = "/"
	}
	// v3 default: <array>/c/i0/i1/.../iN with sep between indices
	// v3 v2-compat: <array>/<i0><sep><i1>...
	var prefix string
	switch enc.Name {
	case "v2":
		prefix = arrayPath + "/"
	default: // "default" or empty
		prefix = arrayPath + "/c" + sep
	}
	out := make([]byte, 0, len(prefix)+8*len(coord))
	out = append(out, prefix...)
	for i, c := range coord {
		if i > 0 {
			out = append(out, sep...)
		}
		out = appendInt(out, c)
	}
	return string(out)
}

func appendInt(b []byte, n int64) []byte {
	if n == 0 {
		return append(b, '0')
	}
	var tmp [20]byte
	i := len(tmp)
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	for n > 0 {
		i--
		tmp[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		tmp[i] = '-'
	}
	return append(b, tmp[i:]...)
}
