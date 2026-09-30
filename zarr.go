// Package zarr is a pure-Go implementation of the Zarr v2 (read) and v3
// (read/write) chunked, compressed, N-dimensional array format.
//
// See the design document for the rationale; this file holds the top-level
// public API surface (Open, OpenArray, Group).
package zarr

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/exergy-dev/zarr-go/codec"
	"github.com/exergy-dev/zarr-go/internal/metadata"
	"github.com/exergy-dev/zarr-go/store"
)

// Node is the union of Group and *Array[T] returned by Open.
//
// Callers typically use OpenArray[T] when the dtype is known at compile time
// and avoid the type-switch entirely.
type Node interface {
	zarrNode()
}

// Group is a v3 group node.
type Group struct {
	store store.Store
	path  string
	meta  *metadata.GroupMeta
}

func (*Group) zarrNode() {}

func (a *Array[T]) zarrNode() {}

// Path returns the group's store path.
func (g *Group) Path() string { return g.path }

// Attrs returns the group's user attributes.
func (g *Group) Attrs() map[string]any { return cloneAttrs(g.meta.Attributes) }

// wrapCorrupt tags a metadata decode failure with ErrCorruptMetadata so
// callers can errors.Is against it, while keeping the underlying error in
// the chain for errors.Is/As.
func wrapCorrupt(err error) error {
	return fmt.Errorf("%w: %w", ErrCorruptMetadata, err)
}

// Open returns the Node (Group or Array) at path. Detects v3 by reading
// zarr.json; falls back to v2 (.zarray / .zgroup) if zarr.json is absent.
func Open(ctx context.Context, s store.Store, path string, opts ...OpenOption) (Node, error) {
	path = strings.Trim(path, "/")
	o := defaultOptions()
	for _, opt := range opts {
		opt(&o)
	}
	am, gm, err := loadMeta(ctx, s, path)
	if err != nil {
		return nil, err
	}
	if gm != nil {
		return &Group{store: s, path: path, meta: gm}, nil
	}
	return openArrayDynamic(s, path, am, o)
}

// OpenArray opens the array at path and asserts its dtype matches T. Detects
// v3 first; falls back to v2 if zarr.json is absent.
func OpenArray[T Numeric](ctx context.Context, s store.Store, path string, opts ...OpenOption) (*Array[T], error) {
	path = strings.Trim(path, "/")
	o := defaultOptions()
	for _, opt := range opts {
		opt(&o)
	}
	am, _, err := loadMeta(ctx, s, path)
	if err != nil {
		return nil, err
	}
	if am == nil {
		return nil, fmt.Errorf("node at %q is not an array", path)
	}
	dt, pipe, fill, err := resolveArray(am)
	if err != nil {
		return nil, err
	}
	if want := DtypeOf[T](); want != dt {
		return nil, fmt.Errorf("dtype mismatch: array is %s, T is %s", dt, want)
	}
	return newTypedArray[T](s, path, am, pipe, dt, fill, o), nil
}

// loadMeta reads the metadata of the node at path: v3 zarr.json first, else
// v2 .zarray or .zgroup plus the optional .zattrs sidecar. On success exactly
// one of the returned array and group metadata is non-nil.
func loadMeta(ctx context.Context, s store.Store, path string) (*metadata.ArrayMeta, *metadata.GroupMeta, error) {
	raw, found, err := getMeta(ctx, s, path, "zarr.json")
	if err != nil {
		return nil, nil, err
	}
	if found {
		am, gm, err := decodeV3(raw)
		if err != nil {
			return nil, nil, wrapCorrupt(err)
		}
		return am, gm, nil
	}
	var (
		am *metadata.ArrayMeta
		gm *metadata.GroupMeta
	)
	if raw, found, err = getMeta(ctx, s, path, ".zarray"); err != nil {
		return nil, nil, err
	} else if found {
		am, err = metadata.DecodeArrayV2(raw)
	} else if raw, found, err = getMeta(ctx, s, path, ".zgroup"); err != nil {
		return nil, nil, err
	} else if found {
		gm, err = metadata.DecodeGroupV2(raw)
	} else {
		return nil, nil, fmt.Errorf("no v2 or v3 metadata found at %q", path)
	}
	if err != nil {
		return nil, nil, wrapCorrupt(err)
	}
	attrs, err := loadAttrs(ctx, s, path)
	if err != nil {
		return nil, nil, err
	}
	if am != nil {
		am.Attributes = attrs
	} else {
		gm.Attributes = attrs
	}
	return am, gm, nil
}

// decodeV3 decodes a zarr.json document into array or group metadata.
func decodeV3(raw []byte) (*metadata.ArrayMeta, *metadata.GroupMeta, error) {
	nt, err := metadata.PeekNodeType(raw)
	if err != nil {
		return nil, nil, err
	}
	switch nt {
	case metadata.NodeArray:
		am, err := metadata.DecodeArray(raw)
		return am, nil, err
	case metadata.NodeGroup:
		gm, err := metadata.DecodeGroup(raw)
		return nil, gm, err
	}
	return nil, nil, errors.New("unknown node type")
}

// getMeta reads the metadata document name at path. A missing key returns
// found == false and no error; any other read error is surfaced with path
// context.
func getMeta(ctx context.Context, s store.Store, path, name string) (raw []byte, found bool, err error) {
	raw, err = s.Get(ctx, joinKey(path, name))
	if errors.Is(err, store.ErrKeyNotFound) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("read %s at %q: %w", name, path, err)
	}
	return raw, true, nil
}

// loadAttrs fetches and parses the optional .zattrs sidecar at path. A
// missing file returns (nil, nil); any other read or parse error is
// surfaced with path context.
func loadAttrs(ctx context.Context, s store.Store, path string) (map[string]any, error) {
	raw, found, err := getMeta(ctx, s, path, ".zattrs")
	if err != nil || !found {
		return nil, err
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("parse .zattrs at %q: %w", path, err)
	}
	return out, nil
}

// resolveArray parses am's dtype, builds its codec pipeline, and decodes its
// fill value: everything an Array needs beyond the metadata itself.
func resolveArray(am *metadata.ArrayMeta) (Dtype, *codec.Pipeline, []byte, error) {
	dt, err := ParseDtype(am.DataType)
	if err != nil {
		return DtypeInvalid, nil, nil, err
	}
	pipe, err := codec.BuildPipeline(am.Codecs)
	if err != nil {
		return DtypeInvalid, nil, nil, err
	}
	fill, err := parseFillValue(am.FillValue, dt)
	if err != nil {
		return DtypeInvalid, nil, nil, fmt.Errorf("parse fill_value: %w", err)
	}
	return dt, pipe, fill, nil
}

// probeMmap returns a non-nil MmapStore if s implements it, else nil.
func probeMmap(s store.Store) store.MmapStore {
	if m, ok := s.(store.MmapStore); ok {
		return m
	}
	return nil
}

// newTypedArray builds an Array[T] from already-resolved metadata, pipeline,
// and dtype. Every Array[T] field is independent of T itself (T only selects
// which concrete instantiation of the generic struct is returned), so this
// one function backs OpenArray, every openArrayDynamic case, and CreateArray.
func newTypedArray[T Numeric](s store.Store, path string, am *metadata.ArrayMeta, pipe *codec.Pipeline, dt Dtype, fill []byte, o arrayOptions) *Array[T] {
	return &Array[T]{
		store:          s,
		mmap:           probeMmap(s),
		path:           path,
		meta:           am,
		pipeline:       pipe,
		dtype:          dt,
		fill:           fill,
		opts:           o,
		rawCopyFast:    isRawCopyFastPipeline(pipe),
		partialDecoder: pipelinePartialDecoder(pipe),
	}
}

// openArrayDynamic returns a Node-typed array whose concrete T matches the
// metadata dtype. Caller can type-switch.
func openArrayDynamic(s store.Store, path string, am *metadata.ArrayMeta, o arrayOptions) (Node, error) {
	dt, pipe, fill, err := resolveArray(am)
	if err != nil {
		return nil, err
	}
	switch dt {
	case DtypeBool:
		return newTypedArray[bool](s, path, am, pipe, dt, fill, o), nil
	case DtypeInt8:
		return newTypedArray[int8](s, path, am, pipe, dt, fill, o), nil
	case DtypeInt16:
		return newTypedArray[int16](s, path, am, pipe, dt, fill, o), nil
	case DtypeInt32:
		return newTypedArray[int32](s, path, am, pipe, dt, fill, o), nil
	case DtypeInt64:
		return newTypedArray[int64](s, path, am, pipe, dt, fill, o), nil
	case DtypeUint8:
		return newTypedArray[uint8](s, path, am, pipe, dt, fill, o), nil
	case DtypeUint16:
		return newTypedArray[uint16](s, path, am, pipe, dt, fill, o), nil
	case DtypeUint32:
		return newTypedArray[uint32](s, path, am, pipe, dt, fill, o), nil
	case DtypeUint64:
		return newTypedArray[uint64](s, path, am, pipe, dt, fill, o), nil
	case DtypeFloat16:
		return newTypedArray[Float16](s, path, am, pipe, dt, fill, o), nil
	case DtypeFloat32:
		return newTypedArray[float32](s, path, am, pipe, dt, fill, o), nil
	case DtypeFloat64:
		return newTypedArray[float64](s, path, am, pipe, dt, fill, o), nil
	case DtypeComplex64:
		return newTypedArray[complex64](s, path, am, pipe, dt, fill, o), nil
	case DtypeComplex128:
		return newTypedArray[complex128](s, path, am, pipe, dt, fill, o), nil
	}
	return nil, fmt.Errorf("%w: %s", ErrUnsupportedDtype, dt)
}

func joinKey(parts ...string) string {
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.Trim(p, "/")
		if p != "" {
			out = append(out, p)
		}
	}
	return strings.Join(out, "/")
}
