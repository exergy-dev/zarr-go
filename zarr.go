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

	"github.com/rkm/zarr-go/codec"
	"github.com/rkm/zarr-go/internal/metadata"
	"github.com/rkm/zarr-go/store"
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
	opts  arrayOptions
}

func (*Group) zarrNode() {}

func (a *Array[T]) zarrNode() {}

// Path returns the group's store path.
func (g *Group) Path() string { return g.path }

// Attrs returns the group's user attributes.
func (g *Group) Attrs() map[string]any { return cloneAttrs(g.meta.Attributes) }

// wrapCorrupt tags a metadata decode failure with ErrCorruptMetadata so
// callers can errors.Is against it, while preserving the underlying message.
func wrapCorrupt(err error) error {
	return fmt.Errorf("%w: %v", ErrCorruptMetadata, err)
}

// Open returns the Node (Group or Array) at path. Detects v3 by reading
// zarr.json; falls back to v2 (.zarray / .zgroup) if zarr.json is absent.
func Open(ctx context.Context, s store.Store, path string, opts ...OpenOption) (Node, error) {
	path = strings.Trim(path, "/")
	o := defaultOptions()
	for _, opt := range opts {
		opt(&o)
	}
	if raw, err := s.Get(ctx, joinKey(path, "zarr.json")); err == nil {
		return openV3(ctx, s, path, raw, o)
	} else if !errors.Is(err, store.ErrKeyNotFound) {
		return nil, fmt.Errorf("read zarr.json at %q: %w", path, err)
	}
	return openV2(ctx, s, path, o)
}

func openV3(ctx context.Context, s store.Store, path string, raw []byte, o arrayOptions) (Node, error) {
	nt, err := metadata.PeekNodeType(raw)
	if err != nil {
		return nil, wrapCorrupt(err)
	}
	switch nt {
	case metadata.NodeGroup:
		gm, err := metadata.DecodeGroup(raw)
		if err != nil {
			return nil, wrapCorrupt(err)
		}
		return &Group{store: s, path: path, meta: gm, opts: o}, nil
	case metadata.NodeArray:
		am, err := metadata.DecodeArray(raw)
		if err != nil {
			return nil, wrapCorrupt(err)
		}
		return openArrayDynamic(s, path, am, o)
	}
	return nil, errors.New("unknown node type")
}

func openV2(ctx context.Context, s store.Store, path string, o arrayOptions) (Node, error) {
	if raw, err := s.Get(ctx, joinKey(path, ".zarray")); err == nil {
		am, err := metadata.DecodeArrayV2(raw)
		if err != nil {
			return nil, wrapCorrupt(err)
		}
		attrs, err := loadAttrs(ctx, s, path)
		if err != nil {
			return nil, err
		}
		am.Attributes = attrs
		return openArrayDynamic(s, path, am, o)
	} else if !errors.Is(err, store.ErrKeyNotFound) {
		return nil, fmt.Errorf("read .zarray at %q: %w", path, err)
	}
	if raw, err := s.Get(ctx, joinKey(path, ".zgroup")); err == nil {
		gm, err := metadata.DecodeGroupV2(raw)
		if err != nil {
			return nil, wrapCorrupt(err)
		}
		attrs, err := loadAttrs(ctx, s, path)
		if err != nil {
			return nil, err
		}
		gm.Attributes = attrs
		return &Group{store: s, path: path, meta: gm, opts: o}, nil
	} else if !errors.Is(err, store.ErrKeyNotFound) {
		return nil, fmt.Errorf("read .zgroup at %q: %w", path, err)
	}
	return nil, fmt.Errorf("no v2 or v3 metadata found at %q", path)
}

func loadArrayMeta(ctx context.Context, s store.Store, path string) (*metadata.ArrayMeta, error) {
	if raw, err := s.Get(ctx, joinKey(path, "zarr.json")); err == nil {
		nt, perr := metadata.PeekNodeType(raw)
		if perr != nil {
			return nil, wrapCorrupt(perr)
		}
		if nt != metadata.NodeArray {
			return nil, fmt.Errorf("node at %q is not an array", path)
		}
		am, derr := metadata.DecodeArray(raw)
		if derr != nil {
			return nil, wrapCorrupt(derr)
		}
		return am, nil
	} else if !errors.Is(err, store.ErrKeyNotFound) {
		return nil, fmt.Errorf("read zarr.json at %q: %w", path, err)
	}
	if raw, err := s.Get(ctx, joinKey(path, ".zarray")); err == nil {
		am, derr := metadata.DecodeArrayV2(raw)
		if derr != nil {
			return nil, wrapCorrupt(derr)
		}
		attrs, lerr := loadAttrs(ctx, s, path)
		if lerr != nil {
			return nil, lerr
		}
		am.Attributes = attrs
		return am, nil
	} else if !errors.Is(err, store.ErrKeyNotFound) {
		return nil, fmt.Errorf("read .zarray at %q: %w", path, err)
	}
	return nil, fmt.Errorf("no v2 or v3 array metadata found at %q", path)
}

// loadAttrs fetches and parses the optional .zattrs sidecar at path. A
// missing file returns (nil, nil); any other read or parse error is
// surfaced with path context.
func loadAttrs(ctx context.Context, s store.Store, path string) (map[string]any, error) {
	raw, err := s.Get(ctx, joinKey(path, ".zattrs"))
	if errors.Is(err, store.ErrKeyNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read .zattrs at %q: %w", path, err)
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("parse .zattrs at %q: %w", path, err)
	}
	return out, nil
}

// OpenArray opens the array at path and asserts its dtype matches T. Detects
// v3 first; falls back to v2 if zarr.json is absent.
func OpenArray[T Numeric](ctx context.Context, s store.Store, path string, opts ...OpenOption) (*Array[T], error) {
	path = strings.Trim(path, "/")
	o := defaultOptions()
	for _, opt := range opts {
		opt(&o)
	}
	am, err := loadArrayMeta(ctx, s, path)
	if err != nil {
		return nil, err
	}
	dt, err := ParseDtype(am.DataType)
	if err != nil {
		return nil, err
	}
	want := DtypeOf[T]()
	if want != dt {
		return nil, fmt.Errorf("dtype mismatch: array is %s, T is %s", dt, want)
	}
	pipe, err := codec.BuildPipeline(am.Codecs)
	if err != nil {
		return nil, err
	}
	fill, err := parseFillValue(am.FillValue, dt)
	if err != nil {
		return nil, fmt.Errorf("parse fill_value: %w", err)
	}
	return newTypedArray[T](s, path, am, pipe, dt, fill, o), nil
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
		fetcher:        newChunkFetcher(s),
	}
}

// openArrayDynamic returns a Node-typed array whose concrete T matches the
// metadata dtype. Caller can type-switch.
func openArrayDynamic(s store.Store, path string, am *metadata.ArrayMeta, o arrayOptions) (Node, error) {
	dt, err := ParseDtype(am.DataType)
	if err != nil {
		return nil, err
	}
	pipe, err := codec.BuildPipeline(am.Codecs)
	if err != nil {
		return nil, err
	}
	fill, err := parseFillValue(am.FillValue, dt)
	if err != nil {
		return nil, fmt.Errorf("parse fill_value: %w", err)
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
