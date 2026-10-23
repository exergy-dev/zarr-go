package zarr_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"sync/atomic"
	"testing"

	zarr "github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/codec"
	"github.com/exergy-dev/zarr-go/codec/crc32c"
	"github.com/exergy-dev/zarr-go/codec/gzip"
	"github.com/exergy-dev/zarr-go/codec/sharding"
	"github.com/exergy-dev/zarr-go/codec/transpose"
	"github.com/exergy-dev/zarr-go/codec/zstd"
	"github.com/exergy-dev/zarr-go/store"
	"github.com/exergy-dev/zarr-go/store/mem"
)

// codecNames returns the codec names in the stored zarr.json for the array
// at path, in pipeline order.
func codecNames(t *testing.T, s store.Store, path string) []string {
	t.Helper()
	raw, err := s.Get(context.Background(), path+"/zarr.json")
	if err != nil {
		t.Fatal(err)
	}
	var meta struct {
		Codecs []codec.Spec `json:"codecs"`
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	names := make([]string, len(meta.Codecs))
	for i, c := range meta.Codecs {
		names[i] = c.Name
	}
	return names
}

func TestWithCodecsInsertsBytesCodec(t *testing.T) {
	ctx := context.Background()
	cases := []struct {
		name  string
		specs []codec.Spec
		want  string
	}{
		{"default", nil, "bytes"},
		{"compressor only", []codec.Spec{zstd.Spec(3)}, "bytes,zstd"},
		{"compressor and checksum", []codec.Spec{gzip.Spec(5), crc32c.Spec()}, "bytes,gzip,crc32c"},
		{"transpose then compressor", []codec.Spec{transpose.Spec(1, 0), gzip.Spec(1)}, "transpose,bytes,gzip"},
		{"transpose only", []codec.Spec{transpose.Spec(1, 0)}, "transpose,bytes"},
		{"explicit sharding", []codec.Spec{sharding.Spec([]int64{2, 2}, zstd.Spec(1))}, "sharding_indexed"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := mem.New()
			shape, chunks := []int64{6, 4}, []int64{4, 4}
			a, err := zarr.CreateArray[int32](ctx, s, "a", shape, chunks, zarr.WithCodecs(tc.specs...))
			if err != nil {
				t.Fatalf("CreateArray: %v", err)
			}
			if got := strings.Join(codecNames(t, s, "a"), ","); got != tc.want {
				t.Fatalf("codecs = %s, want %s", got, tc.want)
			}
			src := make([]int32, 24)
			for i := range src {
				src[i] = int32(i) - 7
			}
			if err := a.Write(ctx, zarr.All(shape), src); err != nil {
				t.Fatalf("Write: %v", err)
			}
			b, err := zarr.OpenArray[int32](ctx, s, "a")
			if err != nil {
				t.Fatalf("OpenArray: %v", err)
			}
			got := make([]int32, 24)
			if err := b.Read(ctx, zarr.All(shape), got); err != nil {
				t.Fatalf("Read: %v", err)
			}
			for i := range src {
				if got[i] != src[i] {
					t.Fatalf("[%d] = %d, want %d", i, got[i], src[i])
				}
			}
		})
	}
}

func TestWithCodecsUnknownCodec(t *testing.T) {
	_, err := zarr.CreateArray[int32](context.Background(), mem.New(), "a", []int64{4}, []int64{2},
		zarr.WithCodecs(codec.Spec{Name: "nope"}))
	if !errors.Is(err, zarr.ErrUnknownCodec) {
		t.Fatalf("err = %v, want ErrUnknownCodec", err)
	}
}

func TestShardingSpecInnerCodecs(t *testing.T) {
	spec := sharding.Spec([]int64{2, 2}, zstd.Spec(1))
	var cfg struct {
		ChunkShape []int64      `json:"chunk_shape"`
		Codecs     []codec.Spec `json:"codecs"`
	}
	if err := json.Unmarshal(spec.Configuration, &cfg); err != nil {
		t.Fatal(err)
	}
	if len(cfg.Codecs) != 2 || cfg.Codecs[0].Name != "bytes" || cfg.Codecs[1].Name != "zstd" {
		t.Fatalf("inner codecs = %+v, want bytes then zstd", cfg.Codecs)
	}
	if len(cfg.ChunkShape) != 2 || cfg.ChunkShape[0] != 2 {
		t.Fatalf("chunk_shape = %v", cfg.ChunkShape)
	}
}

func TestWithFillValueTyped(t *testing.T) {
	ctx := context.Background()
	fillJSON := func(t *testing.T, s store.Store) string {
		raw, err := s.Get(ctx, "a/zarr.json")
		if err != nil {
			t.Fatal(err)
		}
		var meta struct {
			FillValue json.RawMessage `json:"fill_value"`
		}
		if err := json.Unmarshal(raw, &meta); err != nil {
			t.Fatal(err)
		}
		var compact bytes.Buffer
		if err := json.Compact(&compact, meta.FillValue); err != nil {
			t.Fatal(err)
		}
		return compact.String()
	}

	t.Run("float32 NaN", func(t *testing.T) {
		s := mem.New()
		a, err := zarr.CreateArray[float32](ctx, s, "a", []int64{3}, []int64{2},
			zarr.WithFillValue(float32(math.NaN())))
		if err != nil {
			t.Fatal(err)
		}
		if got := fillJSON(t, s); got != `"NaN"` {
			t.Fatalf("fill_value = %s, want \"NaN\"", got)
		}
		if v := a.FillValue(); v == v { // NaN != NaN
			t.Fatalf("FillValue() = %v, want NaN", v)
		}
		got := make([]float32, 3)
		if err := a.Read(ctx, zarr.All([]int64{3}), got); err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if v == v {
				t.Fatalf("[%d] = %v, want NaN", i, v)
			}
		}
	})
	t.Run("float64 -Inf", func(t *testing.T) {
		s := mem.New()
		a, err := zarr.CreateArray[float64](ctx, s, "a", []int64{1}, []int64{1},
			zarr.WithFillValue(math.Inf(-1)))
		if err != nil {
			t.Fatal(err)
		}
		if got := fillJSON(t, s); got != `"-Infinity"` {
			t.Fatalf("fill_value = %s", got)
		}
		if a.FillValue() != math.Inf(-1) {
			t.Fatalf("FillValue() = %v", a.FillValue())
		}
	})
	t.Run("untyped int for float32", func(t *testing.T) {
		s := mem.New()
		a, err := zarr.CreateArray[float32](ctx, s, "a", []int64{1}, []int64{1}, zarr.WithFillValue(-9999))
		if err != nil {
			t.Fatal(err)
		}
		if got := fillJSON(t, s); got != "-9999" {
			t.Fatalf("fill_value = %s", got)
		}
		if a.FillValue() != -9999 {
			t.Fatalf("FillValue() = %v", a.FillValue())
		}
	})
	t.Run("int16", func(t *testing.T) {
		s := mem.New()
		a, err := zarr.CreateArray[int16](ctx, s, "a", []int64{1}, []int64{1}, zarr.WithFillValue(int16(-5)))
		if err != nil {
			t.Fatal(err)
		}
		if a.FillValue() != -5 {
			t.Fatalf("FillValue() = %v", a.FillValue())
		}
	})
	t.Run("bool", func(t *testing.T) {
		s := mem.New()
		a, err := zarr.CreateArray[bool](ctx, s, "a", []int64{1}, []int64{1}, zarr.WithFillValue(true))
		if err != nil {
			t.Fatal(err)
		}
		if got := fillJSON(t, s); got != "true" {
			t.Fatalf("fill_value = %s", got)
		}
		if !a.FillValue() {
			t.Fatal("FillValue() = false")
		}
	})
	t.Run("complex64", func(t *testing.T) {
		s := mem.New()
		a, err := zarr.CreateArray[complex64](ctx, s, "a", []int64{1}, []int64{1},
			zarr.WithFillValue(complex(float32(1.5), float32(math.Inf(1)))))
		if err != nil {
			t.Fatal(err)
		}
		if got := fillJSON(t, s); got != `[1.5,"Infinity"]` {
			t.Fatalf("fill_value = %s", got)
		}
		if v := a.FillValue(); real(v) != 1.5 || !math.IsInf(float64(imag(v)), 1) {
			t.Fatalf("FillValue() = %v", v)
		}
	})
	t.Run("Float16", func(t *testing.T) {
		s := mem.New()
		a, err := zarr.CreateArray[zarr.Float16](ctx, s, "a", []int64{1}, []int64{1},
			zarr.WithFillValue(zarr.Float16(0x3c00))) // 1.0
		if err != nil {
			t.Fatal(err)
		}
		if got := fillJSON(t, s); got != "1" {
			t.Fatalf("fill_value = %s", got)
		}
		if a.FillValue() != 0x3c00 {
			t.Fatalf("FillValue() = %#x", a.FillValue())
		}
	})
	t.Run("raw JSON", func(t *testing.T) {
		s := mem.New()
		a, err := zarr.CreateArray[float32](ctx, s, "a", []int64{1}, []int64{1},
			zarr.WithFillValue(json.RawMessage(`"0x7fc00000"`)))
		if err != nil {
			t.Fatal(err)
		}
		if got := fillJSON(t, s); got != `"0x7fc00000"` {
			t.Fatalf("fill_value = %s", got)
		}
		if v := a.FillValue(); v == v {
			t.Fatalf("FillValue() = %v, want NaN", v)
		}
	})
	t.Run("out of range", func(t *testing.T) {
		_, err := zarr.CreateArray[uint8](ctx, mem.New(), "a", []int64{1}, []int64{1}, zarr.WithFillValue(-1))
		if err == nil || !strings.Contains(err.Error(), "out of range") {
			t.Fatalf("err = %v, want out-of-range", err)
		}
	})
	t.Run("unsupported type", func(t *testing.T) {
		_, err := zarr.CreateArray[uint8](ctx, mem.New(), "a", []int64{1}, []int64{1}, zarr.WithFillValue("zero"))
		if err == nil || !strings.Contains(err.Error(), "unsupported type") {
			t.Fatalf("err = %v, want unsupported type", err)
		}
	})
}

func TestArrayPath(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	a, err := zarr.CreateArray[int8](ctx, s, "/g/a/", []int64{1}, []int64{1})
	if err != nil {
		t.Fatal(err)
	}
	if a.Path() != "g/a" {
		t.Fatalf("Path() = %q, want g/a", a.Path())
	}
	root, err := zarr.CreateArray[int8](ctx, s, "", []int64{1}, []int64{1})
	if err != nil {
		t.Fatal(err)
	}
	if root.Path() != "" {
		t.Fatalf("Path() = %q, want \"\"", root.Path())
	}
}

func TestSelectionZeroStep(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	shape, chunks := []int64{7, 5}, []int64{3, 2}
	a, err := zarr.CreateArray[int32](ctx, s, "a", shape, chunks)
	if err != nil {
		t.Fatal(err)
	}
	src := make([]int32, 35)
	for i := range src {
		src[i] = int32(i)
	}
	// Step 0 on the write side too.
	if err := a.Write(ctx, zarr.Selection{{Start: 0, Stop: 7}, {Start: 0, Stop: 5}}, src); err != nil {
		t.Fatalf("Write: %v", err)
	}
	zero := zarr.Selection{{Start: 1, Stop: 6}, {Start: 2, Stop: 5}}
	one := zarr.Selection{{Start: 1, Stop: 6, Step: 1}, {Start: 2, Stop: 5, Step: 1}}
	if zero.NumElements() != one.NumElements() {
		t.Fatalf("NumElements: %d vs %d", zero.NumElements(), one.NumElements())
	}
	got0 := make([]int32, zero.NumElements())
	got1 := make([]int32, one.NumElements())
	if err := a.Read(ctx, zero, got0); err != nil {
		t.Fatalf("Read(step 0): %v", err)
	}
	if err := a.Read(ctx, one, got1); err != nil {
		t.Fatalf("Read(step 1): %v", err)
	}
	for i := range got0 {
		if got0[i] != got1[i] {
			t.Fatalf("[%d] step0=%d step1=%d", i, got0[i], got1[i])
		}
	}
	var n int
	for c, err := range a.Stream(ctx, zero) {
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		n += len(c.Data)
	}
	if n == 0 {
		t.Fatal("Stream(step 0) yielded no data")
	}
	if err := zero.Validate(shape); err != nil {
		t.Fatalf("Validate(step 0): %v", err)
	}
	neg := zarr.Selection{{Start: 0, Stop: 7, Step: -1}, {Start: 0, Stop: 5}}
	if err := neg.Validate(shape); err == nil {
		t.Fatal("Validate(step -1) = nil, want error")
	}
}

// failingWriter fails every Set after the first n.
type failingWriter struct {
	store.Writer
	allow atomic.Int64
}

var errSetFailed = errors.New("set failed")

func (f *failingWriter) Set(ctx context.Context, key string, value []byte) error {
	if f.allow.Add(-1) < 0 {
		return errSetFailed
	}
	return f.Writer.Set(ctx, key, value)
}

func TestWriteConcurrentRMWAndErrors(t *testing.T) {
	ctx := context.Background()
	shape, chunks := []int64{64, 64}, []int64{8, 8} // 64 chunks
	base := mem.New()
	a, err := zarr.CreateArray[int32](ctx, base, "a", shape, chunks, zarr.WithCodecs(gzip.Spec(1)),
		zarr.WithConcurrency(3), zarr.WithIOConcurrency(5))
	if err != nil {
		t.Fatal(err)
	}
	full := make([]int32, 64*64)
	for i := range full {
		full[i] = int32(i)
	}
	if err := a.Write(ctx, zarr.All(shape), full); err != nil {
		t.Fatalf("Write(all): %v", err)
	}
	// A strided patch touches every chunk partially, so each is RMW'd
	// concurrently.
	patch := zarr.Selection{{Start: 1, Stop: 64, Step: 2}, {Start: 1, Stop: 64, Step: 2}}
	pv := make([]int32, patch.NumElements())
	for i := range pv {
		pv[i] = -int32(i) - 1
	}
	if err := a.Write(ctx, patch, pv); err != nil {
		t.Fatalf("Write(patch): %v", err)
	}
	got := make([]int32, 64*64)
	if err := a.Read(ctx, zarr.All(shape), got); err != nil {
		t.Fatalf("Read: %v", err)
	}
	k := 0
	for r := int64(0); r < 64; r++ {
		for c := int64(0); c < 64; c++ {
			want := full[r*64+c]
			if r%2 == 1 && c%2 == 1 {
				want = pv[k]
				k++
			}
			if got[r*64+c] != want {
				t.Fatalf("(%d,%d) = %d, want %d", r, c, got[r*64+c], want)
			}
		}
	}

	// A failing store surfaces the error (not a silent partial write).
	fw := &failingWriter{Writer: mem.New()}
	fw.allow.Store(1 << 30)
	b, err := zarr.CreateArray[int32](ctx, fw, "b", shape, chunks)
	if err != nil {
		t.Fatal(err)
	}
	fw.allow.Store(10)
	if err := b.Write(ctx, zarr.All(shape), full); !errors.Is(err, errSetFailed) {
		t.Fatalf("Write with failing store: %v, want errSetFailed", err)
	}

	// A cancelled context is reported even if every chunk happened to finish.
	cctx, cancel := context.WithCancel(ctx)
	cancel()
	if err := a.Write(cctx, zarr.All(shape), full); !errors.Is(err, context.Canceled) {
		t.Fatalf("Write(cancelled) = %v, want context.Canceled", err)
	}
}
