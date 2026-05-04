package zarr

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/rkm/zarr-go/codec"
	"github.com/rkm/zarr-go/internal/metadata"
	"github.com/rkm/zarr-go/store"
)

// CreateOption customizes array/group creation.
type CreateOption func(*createOptions)

type createOptions struct {
	codecs     []codec.Spec
	fillValue  json.RawMessage
	attrs      map[string]any
	dimNames   []string
	keyEncName string // "default" or "v2"
	keySep     string
	openOpts   arrayOptions
}

func defaultCreateOptions() createOptions {
	return createOptions{
		codecs: []codec.Spec{{
			Name:          "bytes",
			Configuration: json.RawMessage(`{"endian":"little"}`),
		}},
		keyEncName: "default",
		keySep:     "/",
		openOpts:   defaultOptions(),
	}
}

// WithCodecs sets the codec pipeline used at write time. Callers typically
// supply the bytes codec plus optional compression (e.g., gzip, zstd).
func WithCodecs(specs ...codec.Spec) CreateOption {
	return func(o *createOptions) {
		if len(specs) > 0 {
			o.codecs = append(o.codecs[:0], specs...)
		}
	}
}

// WithFillValue sets the fill value (encoded as v3 JSON: a number, "NaN",
// "Infinity", "-Infinity", or a 2-element array for complex).
func WithFillValue(raw json.RawMessage) CreateOption {
	return func(o *createOptions) {
		o.fillValue = append(json.RawMessage(nil), raw...)
	}
}

// WithAttributes attaches user attributes to the new node.
func WithAttributes(attrs map[string]any) CreateOption {
	return func(o *createOptions) {
		o.attrs = attrs
	}
}

// WithDimensionNames labels the array's dimensions.
func WithDimensionNames(names []string) CreateOption {
	return func(o *createOptions) {
		o.dimNames = append([]string(nil), names...)
	}
}

// CreateArray creates a new v3 array under path with the given shape and chunk
// shape. The store must implement store.WritableStore.
func CreateArray[T Numeric](
	ctx context.Context,
	s store.Store,
	path string,
	shape, chunks []int64,
	opts ...CreateOption,
) (*Array[T], error) {
	ws, ok := s.(store.WritableStore)
	if !ok {
		return nil, fmt.Errorf("store does not support writes")
	}
	if len(shape) == 0 || len(chunks) != len(shape) {
		return nil, fmt.Errorf("shape and chunks must be non-empty and same rank")
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
		opt(&co)
	}
	dt := DtypeOf[T]()
	if dt == DtypeInvalid {
		return nil, fmt.Errorf("%w: T", ErrUnsupportedDtype)
	}

	pipe, err := codec.BuildPipeline(co.codecs)
	if err != nil {
		return nil, err
	}
	if len(co.fillValue) == 0 {
		co.fillValue = json.RawMessage("0")
	}
	fill, err := parseFillValue(co.fillValue, dt)
	if err != nil {
		return nil, fmt.Errorf("parse fill_value: %w", err)
	}

	meta := &metadata.ArrayMeta{
		ZarrFormat: 3,
		NodeType:   "array",
		Shape:      append([]int64(nil), shape...),
		DataType:   dt.String(),
		ChunkGrid: metadata.ChunkGrid{
			Name: "regular",
			Configuration: metadata.ChunkGridRegularCfg{
				ChunkShape: append([]int64(nil), chunks...),
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
	key := joinKey(path, "zarr.json")
	if err := ws.Set(ctx, key, raw); err != nil {
		return nil, fmt.Errorf("write zarr.json: %w", err)
	}

	a := &Array[T]{
		store:       s,
		mmap:        probeMmap(s),
		path:        path,
		meta:        meta,
		pipeline:    pipe,
		dtype:       dt,
		fill:        fill,
		opts:        co.openOpts,
		rawCopyFast: isRawCopyFastPipeline(pipe),
	}
	return a, nil
}

// CreateGroup creates a new v3 group under path. The store must implement
// store.WritableStore.
func CreateGroup(ctx context.Context, s store.Store, path string, attrs map[string]any) (*Group, error) {
	ws, ok := s.(store.WritableStore)
	if !ok {
		return nil, fmt.Errorf("store does not support writes")
	}
	gm := &metadata.GroupMeta{ZarrFormat: 3, NodeType: "group", Attributes: attrs}
	raw, err := metadata.MarshalGroup(gm)
	if err != nil {
		return nil, err
	}
	if err := ws.Set(ctx, joinKey(path, "zarr.json"), raw); err != nil {
		return nil, err
	}
	return &Group{store: s, path: path, meta: gm, opts: defaultOptions()}, nil
}

// CreateGroup adds a sub-group named name under g.
func (g *Group) CreateGroup(ctx context.Context, name string, attrs map[string]any) (*Group, error) {
	return CreateGroup(ctx, g.store, joinKey(g.path, name), attrs)
}

// errReadOnly is returned by writes against non-writable stores.
var errReadOnly = errors.New("store is read-only")
