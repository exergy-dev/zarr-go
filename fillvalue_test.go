package zarr_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strconv"
	"testing"

	"github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/codec"
	"github.com/exergy-dev/zarr-go/store/mem"
)

// TestHexFloatFillValue checks the v3 hex bit-pattern fill_value encoding
// ("0x...") is accepted for float32/float64, decoding to the exact bit
// pattern rather than round-tripping through a lossy float64 conversion.
func TestHexFloatFillValue(t *testing.T) {
	ctx := context.Background()

	t.Run("float32", func(t *testing.T) {
		s := mem.New()
		a, err := zarr.CreateArray[float32](ctx, s, "arr", []int64{4}, []int64{2},
			zarr.WithFillValue(json.RawMessage(`"0x7fc00000"`))) // canonical qNaN
		if err != nil {
			t.Fatalf("CreateArray: %v", err)
		}
		dst := make([]float32, 4)
		if err := a.Read(ctx, zarr.All([]int64{4}), dst); err != nil {
			t.Fatalf("Read: %v", err)
		}
		got := math.Float32bits(dst[0])
		if got != 0x7fc00000 {
			t.Fatalf("fill bits = %#x, want 0x7fc00000", got)
		}
	})

	t.Run("float64", func(t *testing.T) {
		s := mem.New()
		a, err := zarr.CreateArray[float64](ctx, s, "arr", []int64{4}, []int64{2},
			zarr.WithFillValue(json.RawMessage(`"0x7ff8000000000000"`))) // canonical qNaN
		if err != nil {
			t.Fatalf("CreateArray: %v", err)
		}
		dst := make([]float64, 4)
		if err := a.Read(ctx, zarr.All([]int64{4}), dst); err != nil {
			t.Fatalf("Read: %v", err)
		}
		got := math.Float64bits(dst[0])
		if got != 0x7ff8000000000000 {
			t.Fatalf("fill bits = %#x, want 0x7ff8000000000000", got)
		}
	})

	t.Run("invalid hex rejected", func(t *testing.T) {
		s := mem.New()
		_, err := zarr.CreateArray[float32](ctx, s, "arr", []int64{4}, []int64{2},
			zarr.WithFillValue(json.RawMessage(`"0xzz"`)))
		if err == nil {
			t.Fatal("expected error for invalid hex fill value")
		}
	})
}

// TestErrCorruptMetadataWrapsDecodeFailures checks Open/OpenArray wrap
// metadata decode errors with the exported ErrCorruptMetadata sentinel, so
// callers can errors.Is against it instead of string-matching.
func TestErrCorruptMetadataWrapsDecodeFailures(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	// A structurally invalid v3 doc: chunk_shape rank mismatch.
	bad := []byte(`{
		"zarr_format": 3,
		"node_type": "array",
		"shape": [4, 4],
		"data_type": "float32",
		"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
		"fill_value": 0,
		"codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]
	}`)
	if err := s.Set(ctx, "arr/zarr.json", bad); err != nil {
		t.Fatal(err)
	}
	_, err := zarr.OpenArray[float32](ctx, s, "arr")
	if !errors.Is(err, zarr.ErrCorruptMetadata) {
		t.Fatalf("OpenArray: got %v, want ErrCorruptMetadata", err)
	}
	_, err = zarr.Open(ctx, s, "arr")
	if !errors.Is(err, zarr.ErrCorruptMetadata) {
		t.Fatalf("Open: got %v, want ErrCorruptMetadata", err)
	}
}

// TestErrCorruptMetadataKeepsCause checks the ErrCorruptMetadata wrap keeps
// the underlying decode error in the chain, so callers can errors.As for
// the concrete cause (here, the JSON syntax error) as well as the sentinel.
func TestErrCorruptMetadataKeepsCause(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	if err := s.Set(ctx, "arr/zarr.json", []byte(`{"zarr_format": 3,`)); err != nil {
		t.Fatal(err)
	}
	_, openErr := zarr.Open(ctx, s, "arr")
	_, arrErr := zarr.OpenArray[float32](ctx, s, "arr")
	for name, err := range map[string]error{"Open": openErr, "OpenArray": arrErr} {
		if !errors.Is(err, zarr.ErrCorruptMetadata) {
			t.Errorf("%s: got %v, want ErrCorruptMetadata", name, err)
		}
		if !errors.As(err, new(*json.SyntaxError)) {
			t.Errorf("%s: errors.As(%v, *json.SyntaxError) = false, want true", name, err)
		}
	}
}

// TestErrUnknownCodecSentinel checks BuildPipeline's error is wrapped with
// the shared codec.ErrUnknownCodec / zarr.ErrUnknownCodec sentinel (the two
// are aliases), matchable from either package.
func TestErrUnknownCodecSentinel(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	_, err := zarr.CreateArray[float32](ctx, s, "arr", []int64{4}, []int64{2},
		zarr.WithCodecs(codec.Spec{Name: "not_a_real_codec"}))
	if !errors.Is(err, zarr.ErrUnknownCodec) {
		t.Fatalf("got %v, want ErrUnknownCodec", err)
	}
	if !errors.Is(err, codec.ErrUnknownCodec) {
		t.Fatalf("got %v, want codec.ErrUnknownCodec", err)
	}
}

// TestV2LargeIntegerFillValueExact checks v2 integer fill values beyond
// float64's exact range (2^53) read back exactly rather than being
// round-tripped through float64.
func TestV2LargeIntegerFillValueExact(t *testing.T) {
	ctx := context.Background()
	zarray := func(dtype, fill string) []byte {
		return []byte(`{"zarr_format": 2, "shape": [3], "chunks": [2], "dtype": "` + dtype +
			`", "compressor": null, "filters": null, "fill_value": ` + fill + `, "order": "C"}`)
	}

	for _, want := range []int64{math.MaxInt64, math.MinInt64, 1<<53 + 1} {
		s := mem.New()
		if err := s.Set(ctx, "arr/.zarray", zarray("<i8", strconv.FormatInt(want, 10))); err != nil {
			t.Fatal(err)
		}
		a, err := zarr.OpenArray[int64](ctx, s, "arr")
		if err != nil {
			t.Fatalf("OpenArray(fill=%d): %v", want, err)
		}
		dst := make([]int64, 3)
		if err := a.Read(ctx, zarr.All([]int64{3}), dst); err != nil {
			t.Fatalf("Read: %v", err)
		}
		for i, got := range dst {
			if got != want {
				t.Fatalf("int64 fill: dst[%d] = %d, want %d", i, got, want)
			}
		}
	}

	s := mem.New()
	if err := s.Set(ctx, "arr/.zarray", zarray("<u8", "18446744073709551615")); err != nil {
		t.Fatal(err)
	}
	a, err := zarr.OpenArray[uint64](ctx, s, "arr")
	if err != nil {
		t.Fatalf("OpenArray(uint64): %v", err)
	}
	dst := make([]uint64, 3)
	if err := a.Read(ctx, zarr.All([]int64{3}), dst); err != nil {
		t.Fatalf("Read: %v", err)
	}
	for i, got := range dst {
		if got != math.MaxUint64 {
			t.Fatalf("uint64 fill: dst[%d] = %d, want %d", i, got, uint64(math.MaxUint64))
		}
	}
}
