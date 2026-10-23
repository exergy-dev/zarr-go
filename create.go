package zarr

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"github.com/exergy-dev/zarr-go/codec"
	"github.com/exergy-dev/zarr-go/internal/metadata"
	"github.com/exergy-dev/zarr-go/store"
)

// CreateOption customizes CreateArray and CreateGroup. WithAttributes and
// WithOverwrite apply to both; every other option only means something for
// an array, and CreateGroup returns an error naming it rather than silently
// ignoring it.
//
// Every OpenOption is also a CreateOption: passed to CreateArray, it
// configures the returned Array exactly as it would for OpenArray.
type CreateOption interface {
	applyCreate(*createOptions)
}

// createOptionFunc adapts a plain function to CreateOption.
type createOptionFunc func(*createOptions)

func (f createOptionFunc) applyCreate(o *createOptions) { f(o) }

type createOptions struct {
	codecs     []codec.Spec
	fillValue  json.RawMessage
	fillErr    error // a WithFillValue argument that could not be encoded
	attrs      map[string]any
	dimNames   []string
	keyEncName string // "default" or "v2"
	keySep     string
	openOpts   arrayOptions
	overwrite  bool
	// arrayOnly names the last array-only option applied, for CreateGroup
	// to reject.
	arrayOnly string
}

func defaultCreateOptions() createOptions {
	return createOptions{
		codecs:     []codec.Spec{codec.DefaultArrayToBytes},
		keyEncName: "default",
		keySep:     "/",
		openOpts:   defaultOptions(),
	}
}

// WithCodecs sets the array's codec pipeline, in the order the v3 metadata
// lists it: array→array codecs (transpose), then the array→bytes codec
// (bytes or sharding_indexed), then bytes→bytes codecs (compression,
// crc32c). Each codec package has a Spec constructor:
//
//	zarr.WithCodecs(zstd.Spec(3))
//	zarr.WithCodecs(sharding.Spec([]int64{64, 64}, zstd.Spec(3), crc32c.Spec()))
//
// When the list has no array→bytes codec, a little-endian bytes codec is
// inserted in front of the first bytes→bytes codec, so a bare compressor is
// enough. The default pipeline is that bytes codec alone (no compression).
func WithCodecs(specs ...codec.Spec) CreateOption {
	return createOptionFunc(func(o *createOptions) {
		o.arrayOnly = "WithCodecs"
		if len(specs) > 0 {
			o.codecs = append(o.codecs[:0], specs...)
		}
	})
}

// WithFillValue sets the value a chunk that was never written reads as. The
// default is zero (false for bool). v is a Go value of the array's element
// kind: a bool, any integer type, any float type (NaN and ±Inf included),
// Float16, or a complex type; or a json.RawMessage holding the fill value in
// its v3 metadata form, such as a hex bit pattern ("0x7fc00000") for a
// specific NaN payload. The value must be representable in the array's
// dtype: a fractional or out-of-range value is reported by CreateArray.
func WithFillValue(v any) CreateOption {
	return createOptionFunc(func(o *createOptions) {
		o.arrayOnly = "WithFillValue"
		o.fillValue, o.fillErr = encodeFillValue(v)
	})
}

// WithAttributes attaches user attributes to the new node. The map is
// shallow-copied so later mutations by the caller don't alter the node's
// in-memory or already-persisted metadata.
func WithAttributes(attrs map[string]any) CreateOption {
	return createOptionFunc(func(o *createOptions) {
		o.attrs = cloneAttrs(attrs)
	})
}

// WithDimensionNames labels the array's dimensions.
func WithDimensionNames(names []string) CreateOption {
	return createOptionFunc(func(o *createOptions) {
		o.arrayOnly = "WithDimensionNames"
		o.dimNames = append([]string(nil), names...)
	})
}

// WithOverwrite makes CreateArray / CreateGroup unconditionally overwrite
// any existing zarr.json at the target path. Without it, concurrent creates
// use SetIfNotExists and the loser receives store.ErrAlreadyExists.
//
// For CreateArray, an overwrite also deletes every existing key under path
// (chunks included) before the new metadata is written, so re-creating an
// array with a different shape/dtype/chunking at the same path can't leave
// stale chunk bytes that the new metadata's codec pipeline would decode as
// garbage.
func WithOverwrite() CreateOption {
	return createOptionFunc(func(o *createOptions) {
		o.overwrite = true
	})
}

// WithChunkKeySeparator sets the separator used between chunk-coordinate
// components. The default chunk_key_encoding's default separator is "/";
// WithV2ChunkKeys defaults it to ".". Call this after WithV2ChunkKeys in the
// options list to override that default.
func WithChunkKeySeparator(sep string) CreateOption {
	return createOptionFunc(func(o *createOptions) {
		o.arrayOnly = "WithChunkKeySeparator"
		o.keySep = sep
	})
}

// WithV2ChunkKeys selects the v3 "v2"-compatible chunk_key_encoding (e.g.
// "arr/0.1" instead of the default "arr/c/0/1"), matching the key layout
// Zarr v2 stores use, and defaults the separator to "." (v2's convention).
// Call WithChunkKeySeparator after this option to use a different separator.
func WithV2ChunkKeys() CreateOption {
	return createOptionFunc(func(o *createOptions) {
		o.arrayOnly = "WithV2ChunkKeys"
		o.keyEncName = "v2"
		o.keySep = "."
	})
}

// writeMetaAtomic writes a freshly-marshaled zarr.json under key. With
// overwrite=false (the default for Create*) it goes through SetIfNotExists
// so concurrent producers can race safely; the loser surfaces as
// store.ErrAlreadyExists. kind/path are folded into the conflict error.
func writeMetaAtomic(ctx context.Context, ws store.Writer, key string, raw []byte, overwrite bool, kind, path string) error {
	if overwrite {
		if err := ws.Set(ctx, key, raw); err != nil {
			return fmt.Errorf("write zarr.json: %w", err)
		}
		return nil
	}
	ok, err := ws.SetIfNotExists(ctx, key, raw)
	if err != nil {
		return fmt.Errorf("write zarr.json: %w", err)
	}
	if !ok {
		return fmt.Errorf("create %s at %q: %w", kind, path, store.ErrAlreadyExists)
	}
	return nil
}

// CreateArray creates a new v3 array under path with the given shape and chunk
// shape. shape/chunks may both be empty to create a rank-0 (scalar) array.
// The store must implement store.Writer.
func CreateArray[T Numeric](
	ctx context.Context,
	s store.Store,
	path string,
	shape, chunks []int64,
	opts ...CreateOption,
) (*Array[T], error) {
	ws, ok := s.(store.Writer)
	if !ok {
		return nil, ErrReadOnly
	}
	path = strings.Trim(path, "/")
	if len(chunks) != len(shape) {
		return nil, fmt.Errorf("shape and chunks must have the same rank")
	}
	for i, c := range chunks {
		if c <= 0 {
			return nil, fmt.Errorf("chunks[%d] must be positive, got %d", i, c)
		}
		if shape[i] < 0 {
			return nil, fmt.Errorf("shape[%d] must be non-negative, got %d", i, shape[i])
		}
	}
	co := defaultCreateOptions()
	for _, opt := range opts {
		opt.applyCreate(&co)
	}
	if co.fillErr != nil {
		return nil, co.fillErr
	}
	if co.dimNames != nil && len(co.dimNames) != len(shape) {
		return nil, fmt.Errorf("got %d dimension names for a rank-%d array", len(co.dimNames), len(shape))
	}
	dt := DtypeOf[T]()
	if dt == DtypeInvalid {
		return nil, fmt.Errorf("%w: T", ErrUnsupportedDtype)
	}

	codecs, err := codec.EnsureArrayToBytes(co.codecs)
	if err != nil {
		return nil, err
	}
	co.codecs = codecs
	pipe, err := codec.BuildPipeline(co.codecs)
	if err != nil {
		return nil, err
	}
	if len(co.fillValue) == 0 {
		co.fillValue = metadata.DefaultFillValue(dt.String())
	}
	fill, err := parseFillValue(co.fillValue, dt)
	if err != nil {
		return nil, fmt.Errorf("parse fill_value: %w", err)
	}

	meta := &metadata.ArrayMeta{
		ZarrFormat: 3,
		NodeType:   "array",
		Shape:      metadata.CloneInt64(shape),
		DataType:   dt.String(),
		ChunkGrid: metadata.ChunkGrid{
			Name: "regular",
			Configuration: metadata.ChunkGridRegularCfg{
				ChunkShape: metadata.CloneInt64(chunks),
			},
		},
		ChunkKeyEnc: metadata.ChunkKeyEncoding{
			Name: co.keyEncName,
			Configuration: metadata.ChunkKeyEncodingConfig{
				Separator: co.keySep,
			},
		},
		FillValue:  co.fillValue,
		Codecs:     co.codecs,
		Attributes: co.attrs,
		Dimensions: co.dimNames,
	}
	raw, err := metadata.MarshalArray(meta)
	if err != nil {
		return nil, err
	}
	if co.overwrite {
		if err := deleteStaleChunks(ctx, ws, path); err != nil {
			return nil, err
		}
	}
	if err := writeMetaAtomic(ctx, ws, joinKey(path, "zarr.json"), raw, co.overwrite, "array", path); err != nil {
		return nil, err
	}

	return newTypedArray[T](s, path, meta, pipe, dt, fill, co.openOpts), nil
}

// CreateGroup creates a new v3 group under path. The store must implement
// store.Writer. By default this is an atomic create: if a zarr.json
// already exists at the target the call returns store.ErrAlreadyExists.
// Pass WithOverwrite() to clobber an existing entry, and WithAttributes to
// attach user attributes. Any other (array-only) option is an error.
func CreateGroup(ctx context.Context, s store.Store, path string, opts ...CreateOption) (*Group, error) {
	ws, ok := s.(store.Writer)
	if !ok {
		return nil, ErrReadOnly
	}
	path = strings.Trim(path, "/")
	var co createOptions
	for _, opt := range opts {
		opt.applyCreate(&co)
	}
	if co.arrayOnly != "" {
		return nil, fmt.Errorf("create group at %q: %s is an array-only option", path, co.arrayOnly)
	}
	gm := &metadata.GroupMeta{ZarrFormat: 3, NodeType: "group", Attributes: co.attrs}
	raw, err := metadata.MarshalGroup(gm)
	if err != nil {
		return nil, err
	}
	if err := writeMetaAtomic(ctx, ws, joinKey(path, "zarr.json"), raw, co.overwrite, "group", path); err != nil {
		return nil, err
	}
	return &Group{store: s, path: path, meta: gm}, nil
}

// deleteStaleChunks removes every existing key under path (chunks, and any
// stale metadata) before a WithOverwrite() create writes fresh metadata.
// Deleting before writing — rather than after — means a crash mid-cleanup
// leaves the array simply absent (metadata not yet written) instead of
// present with a mix of old and new chunk bytes.
func deleteStaleChunks(ctx context.Context, ws store.Writer, path string) error {
	prefix := path
	if prefix != "" {
		prefix += "/"
	}
	for key, err := range ws.List(ctx, prefix) {
		if err != nil {
			return fmt.Errorf("list existing keys at %q: %w", path, err)
		}
		if err := ws.Delete(ctx, key); err != nil {
			return fmt.Errorf("delete stale key %q: %w", key, err)
		}
	}
	return nil
}

// cloneAttrs returns a shallow copy of attrs (or nil for nil), so callers
// storing a map into metadata don't alias the caller's own map.
func cloneAttrs(attrs map[string]any) map[string]any { return maps.Clone(attrs) }

// CreateGroup adds a sub-group named name under g.
func (g *Group) CreateGroup(ctx context.Context, name string, opts ...CreateOption) (*Group, error) {
	return CreateGroup(ctx, g.store, joinKey(g.path, name), opts...)
}
