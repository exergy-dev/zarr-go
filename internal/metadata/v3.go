// Package metadata parses Zarr v3 array and group JSON.
//
// The exposed types are intentionally narrow — only the fields the engine
// needs. Unknown top-level fields are preserved in ArrayMeta.Extra /
// GroupMeta.Extra so a decode→marshal cycle is lossless even when the
// document contains future-spec fields this implementation doesn't yet
// recognize.
package metadata

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"

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
//
// Extra preserves any top-level v3 fields that this implementation doesn't
// yet recognize, so future spec extensions round-trip losslessly.
type ArrayMeta struct {
	ZarrFormat   int              `json:"zarr_format"`
	NodeType     string           `json:"node_type"`
	Shape        []int64          `json:"shape"`
	DataType     string           `json:"data_type"`
	ChunkGrid    ChunkGrid        `json:"chunk_grid"`
	ChunkKeyEnc  ChunkKeyEncoding `json:"chunk_key_encoding"`
	FillValue    json.RawMessage  `json:"fill_value"`
	Codecs       []codec.Spec     `json:"codecs"`
	Attributes   map[string]any   `json:"attributes,omitempty"`
	StorageTrans json.RawMessage  `json:"storage_transformers,omitempty"`
	Dimensions   []string         `json:"dimension_names,omitempty"`

	// Extra holds top-level fields not covered by the named members. The
	// MarshalJSON path emits them in sorted order alongside the known fields
	// so unknown future-spec keys round-trip losslessly. It is not part of
	// the v3 wire format itself.
	Extra map[string]json.RawMessage `json:"-"`
}

// GroupMeta is the v3 zarr.json for a group.
type GroupMeta struct {
	ZarrFormat int                        `json:"zarr_format"`
	NodeType   string                     `json:"node_type"`
	Attributes map[string]any             `json:"attributes,omitempty"`
	Extra      map[string]json.RawMessage `json:"-"`
}

// ChunkGrid describes how the array is partitioned into chunks. Only the
// "regular" grid is supported in M1.
type ChunkGrid struct {
	Name          string              `json:"name"`
	Configuration ChunkGridRegularCfg `json:"configuration"`
}

type ChunkGridRegularCfg struct {
	ChunkShape []int64 `json:"chunk_shape"`
}

// ChunkKeyEncoding controls how chunk coordinates map to keys in the store.
type ChunkKeyEncoding struct {
	Name          string                 `json:"name"`
	Configuration ChunkKeyEncodingConfig `json:"configuration"`
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

// knownArrayFields lists every JSON key that maps onto a named ArrayMeta
// member. Keys outside this set are preserved in ArrayMeta.Extra.
var knownArrayFields = map[string]struct{}{
	"zarr_format":          {},
	"node_type":            {},
	"shape":                {},
	"data_type":            {},
	"chunk_grid":           {},
	"chunk_key_encoding":   {},
	"fill_value":           {},
	"codecs":               {},
	"attributes":           {},
	"storage_transformers": {},
	"dimension_names":      {},
}

var knownGroupFields = map[string]struct{}{
	"zarr_format": {},
	"node_type":   {},
	"attributes":  {},
}

// DecodeArray parses a zarr.json array document and validates required fields.
func DecodeArray(raw []byte) (*ArrayMeta, error) {
	fields, err := parseTopLevel(raw)
	if err != nil {
		return nil, fmt.Errorf("parse array meta: %w", err)
	}
	m := &ArrayMeta{}
	if err := unmarshalKnown(fields, m, knownArrayFields); err != nil {
		return nil, fmt.Errorf("parse array meta: %w", err)
	}
	for k, v := range fields {
		if _, ok := knownArrayFields[k]; ok {
			continue
		}
		if m.Extra == nil {
			m.Extra = make(map[string]json.RawMessage, 4)
		}
		m.Extra[k] = v
	}
	if m.ZarrFormat != 3 || m.NodeType != "array" {
		return nil, fmt.Errorf("not a v3 array document")
	}
	if len(m.Shape) == 0 {
		return nil, fmt.Errorf("array shape missing")
	}
	for i, d := range m.Shape {
		if d < 0 {
			return nil, fmt.Errorf("shape[%d] must be non-negative, got %d", i, d)
		}
	}
	if m.ChunkGrid.Name != "regular" {
		return nil, fmt.Errorf("unsupported chunk_grid %q (only the regular grid is supported)", m.ChunkGrid.Name)
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
	if err := validateStorageTransformers(m.StorageTrans); err != nil {
		return nil, err
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

// validateStorageTransformers enforces the v3 spec invariant that the
// storage_transformers field MUST be absent, null, or an empty array — no
// standard transformers are defined.
func validateStorageTransformers(raw json.RawMessage) error {
	if len(raw) == 0 {
		return nil
	}
	trimmed := bytes.TrimSpace(raw)
	if bytes.Equal(trimmed, []byte("null")) {
		return nil
	}
	var arr []json.RawMessage
	if err := json.Unmarshal(trimmed, &arr); err != nil {
		return fmt.Errorf("storage_transformers must be null or an array: %w", err)
	}
	if len(arr) > 0 {
		return fmt.Errorf("storage_transformers is non-empty (no standard transformers are defined in Zarr v3)")
	}
	return nil
}

// parseTopLevel decodes raw into a flat key→raw-value map. Used by DecodeArray
// and DecodeGroup to avoid the parse-twice pattern of separately filling a
// typed struct AND a map.
func parseTopLevel(raw []byte) (map[string]json.RawMessage, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, err
	}
	return fields, nil
}

// unmarshalKnown re-marshals only the known fields into a synthetic JSON
// object and unmarshals that into dst. Doing it this way means we parse the
// raw input exactly once (in parseTopLevel) and pay an O(known-fields) JSON
// round-trip on a much smaller buffer instead of re-parsing the whole doc.
func unmarshalKnown(fields map[string]json.RawMessage, dst any, known map[string]struct{}) error {
	subset := make(map[string]json.RawMessage, len(known))
	for k := range known {
		if v, ok := fields[k]; ok {
			subset[k] = v
		}
	}
	buf, err := json.Marshal(subset)
	if err != nil {
		return err
	}
	return json.Unmarshal(buf, dst)
}

// MarshalArray serializes an ArrayMeta into the canonical v3 zarr.json form.
//
// Required fields (zarr_format, node_type, shape, data_type, chunk_grid,
// chunk_key_encoding, fill_value, codecs) are always emitted. Attributes,
// storage_transformers and dimension_names are omitted when empty. Output
// is deterministic: keys at every nesting level are emitted in sorted order
// so two invocations on the same logical metadata produce byte-equal JSON.
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
	// Build the raw object as a sorted map[string]json.RawMessage so the
	// final JSON is reproducible regardless of Go's map-iteration order.
	fields := map[string]json.RawMessage{}
	if err := assignField(fields, "zarr_format", out.ZarrFormat); err != nil {
		return nil, err
	}
	if err := assignField(fields, "node_type", out.NodeType); err != nil {
		return nil, err
	}
	if err := assignField(fields, "shape", out.Shape); err != nil {
		return nil, err
	}
	if err := assignField(fields, "data_type", out.DataType); err != nil {
		return nil, err
	}
	if err := assignField(fields, "chunk_grid", out.ChunkGrid); err != nil {
		return nil, err
	}
	if err := assignField(fields, "chunk_key_encoding", out.ChunkKeyEnc); err != nil {
		return nil, err
	}
	fields["fill_value"] = out.FillValue
	if err := assignField(fields, "codecs", out.Codecs); err != nil {
		return nil, err
	}
	if len(out.Attributes) > 0 {
		raw, err := marshalSorted(out.Attributes)
		if err != nil {
			return nil, fmt.Errorf("marshal attributes: %w", err)
		}
		fields["attributes"] = raw
	}
	if len(out.StorageTrans) > 0 {
		fields["storage_transformers"] = out.StorageTrans
	}
	if len(out.Dimensions) > 0 {
		if err := assignField(fields, "dimension_names", out.Dimensions); err != nil {
			return nil, err
		}
	}
	for k, v := range out.Extra {
		if _, ok := knownArrayFields[k]; ok {
			continue // never let Extra collide with known keys
		}
		fields[k] = v
	}
	return marshalSortedRaw(fields)
}

// MarshalGroup serializes a GroupMeta into the canonical v3 zarr.json form.
// Output is deterministic; unknown top-level fields preserved in Extra
// round-trip alongside the known fields.
func MarshalGroup(m *GroupMeta) ([]byte, error) {
	out := *m
	if out.ZarrFormat == 0 {
		out.ZarrFormat = 3
	}
	if out.NodeType == "" {
		out.NodeType = "group"
	}
	fields := map[string]json.RawMessage{}
	if err := assignField(fields, "zarr_format", out.ZarrFormat); err != nil {
		return nil, err
	}
	if err := assignField(fields, "node_type", out.NodeType); err != nil {
		return nil, err
	}
	if len(out.Attributes) > 0 {
		raw, err := marshalSorted(out.Attributes)
		if err != nil {
			return nil, fmt.Errorf("marshal attributes: %w", err)
		}
		fields["attributes"] = raw
	}
	for k, v := range out.Extra {
		if _, ok := knownGroupFields[k]; ok {
			continue
		}
		fields[k] = v
	}
	return marshalSortedRaw(fields)
}

// assignField marshals v as JSON and stores the result in fields[key].
func assignField(fields map[string]json.RawMessage, key string, v any) error {
	raw, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshal %s: %w", key, err)
	}
	fields[key] = raw
	return nil
}

// marshalSortedRaw emits a JSON object whose keys appear in sorted order, with
// two-space indentation matching json.MarshalIndent's output. Atomic values
// (numbers, strings, bools, null) skip the Compact+Indent round-trip since
// json.Marshal already produces canonical bytes for them.
func marshalSortedRaw(fields map[string]json.RawMessage) ([]byte, error) {
	keys := make([]string, 0, len(fields))
	for k := range fields {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var buf bytes.Buffer
	buf.WriteString("{\n")
	for i, k := range keys {
		keyJSON, err := json.Marshal(k)
		if err != nil {
			return nil, err
		}
		buf.WriteString("  ")
		buf.Write(keyJSON)
		buf.WriteString(": ")
		if err := writeIndented(&buf, fields[k]); err != nil {
			return nil, err
		}
		if i < len(keys)-1 {
			buf.WriteByte(',')
		}
		buf.WriteByte('\n')
	}
	buf.WriteString("}")
	return buf.Bytes(), nil
}

// writeIndented writes v into buf with two-space indentation. For atomic
// JSON values (anything that doesn't open with '{' or '[') it writes the
// trimmed bytes directly; for objects and arrays it does the
// Compact-then-Indent round-trip so embedded whitespace is canonicalized.
func writeIndented(buf *bytes.Buffer, v json.RawMessage) error {
	trimmed := bytes.TrimSpace(v)
	if len(trimmed) == 0 {
		buf.Write(v)
		return nil
	}
	switch trimmed[0] {
	case '{', '[':
		var compact bytes.Buffer
		if err := json.Compact(&compact, v); err != nil {
			return err
		}
		var indented bytes.Buffer
		if err := json.Indent(&indented, compact.Bytes(), "  ", "  "); err != nil {
			return err
		}
		buf.Write(indented.Bytes())
		return nil
	default:
		buf.Write(trimmed)
		return nil
	}
}

// marshalSorted serializes v deterministically. Maps with string keys (at any
// nesting depth) emit keys in sorted order. Non-map values fall through to
// the stdlib encoder.
func marshalSorted(v any) ([]byte, error) {
	switch t := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		var buf bytes.Buffer
		buf.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				buf.WriteByte(',')
			}
			keyJSON, err := json.Marshal(k)
			if err != nil {
				return nil, err
			}
			buf.Write(keyJSON)
			buf.WriteByte(':')
			child, err := marshalSorted(t[k])
			if err != nil {
				return nil, err
			}
			buf.Write(child)
		}
		buf.WriteByte('}')
		return buf.Bytes(), nil
	case []any:
		var buf bytes.Buffer
		buf.WriteByte('[')
		for i, el := range t {
			if i > 0 {
				buf.WriteByte(',')
			}
			child, err := marshalSorted(el)
			if err != nil {
				return nil, err
			}
			buf.Write(child)
		}
		buf.WriteByte(']')
		return buf.Bytes(), nil
	default:
		return json.Marshal(v)
	}
}

// DecodeGroup parses a zarr.json group document.
func DecodeGroup(raw []byte) (*GroupMeta, error) {
	fields, err := parseTopLevel(raw)
	if err != nil {
		return nil, fmt.Errorf("parse group meta: %w", err)
	}
	m := &GroupMeta{}
	if err := unmarshalKnown(fields, m, knownGroupFields); err != nil {
		return nil, fmt.Errorf("parse group meta: %w", err)
	}
	if m.ZarrFormat != 3 || m.NodeType != "group" {
		return nil, fmt.Errorf("not a v3 group document")
	}
	for k, v := range fields {
		if _, ok := knownGroupFields[k]; ok {
			continue
		}
		if m.Extra == nil {
			m.Extra = make(map[string]json.RawMessage, 4)
		}
		m.Extra[k] = v
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
