package zarr_test

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/store"
	"github.com/exergy-dev/zarr-go/store/mem"
)

// v3ArrayDoc returns a minimal v3 array zarr.json with the given dtype,
// chunk shape, fill value, and codecs JSON fragments.
func v3ArrayDoc(dtype, chunks, fill, codecs string) string {
	return `{"zarr_format": 3, "node_type": "array", "shape": [4], "data_type": "` + dtype + `",
		"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": ` + chunks + `}},
		"chunk_key_encoding": {"name": "default"}, "fill_value": ` + fill + `,
		"codecs": ` + codecs + `}`
}

const (
	bytesLE     = `[{"name": "bytes", "configuration": {"endian": "little"}}]`
	v2F4Array   = `{"zarr_format": 2, "shape": [4], "chunks": [2], "dtype": "<f4", "compressor": null, "filters": null, "fill_value": 0, "order": "C"}`
	v3GroupDoc  = `{"zarr_format": 3, "node_type": "group", "attributes": {"k": "v3"}}`
	v2GroupDoc  = `{"zarr_format": 2}`
	v2AttrsDoc  = `{"k": "v2"}`
	corruptJSON = `{"zarr_format": 3, "node_type": `
)

// openWant is the expected outcome of one Open or OpenArray call.
type openWant struct {
	typ  string  // %T of the returned node; "" means an error is expected
	attr string  // for a *zarr.Group: expected Attrs()["k"]
	is   []error // sentinels the error must match (all others must not)
	msg  string  // substring the error must contain
}

var openSentinels = []error{
	zarr.ErrCorruptMetadata,
	zarr.ErrUnknownCodec,
	zarr.ErrUnsupportedDtype,
	store.ErrKeyNotFound,
}

func checkOpen(t *testing.T, what string, node any, err error, w openWant) {
	t.Helper()
	if w.typ != "" {
		if err != nil {
			t.Fatalf("%s: unexpected error %v", what, err)
		}
		if got := fmt.Sprintf("%T", node); got != w.typ {
			t.Fatalf("%s: returned %s, want %s", what, got, w.typ)
		}
		if g, ok := node.(*zarr.Group); ok && g.Attrs()["k"] != w.attr {
			t.Fatalf("%s: Attrs()[k] = %v, want %q", what, g.Attrs()["k"], w.attr)
		}
		return
	}
	if err == nil {
		t.Fatalf("%s: got %T, want error", what, node)
	}
	for _, s := range openSentinels {
		want := false
		for _, is := range w.is {
			want = want || is == s
		}
		if errors.Is(err, s) != want {
			t.Errorf("%s: errors.Is(%v, %v) = %v, want %v", what, err, s, !want, want)
		}
	}
	if !strings.Contains(err.Error(), w.msg) {
		t.Errorf("%s: error %q does not contain %q", what, err, w.msg)
	}
}

// TestOpenErrorMatrix pins the node types and error sentinels Open and
// OpenArray return across v3/v2 arrays and groups and every metadata
// failure mode, so both entry points stay consistent with each other.
func TestOpenErrorMatrix(t *testing.T) {
	corrupt := []error{zarr.ErrCorruptMetadata}
	cases := []struct {
		name      string
		files     map[string]string
		open      openWant
		openArray openWant // OpenArray[float32]
	}{
		{
			name:      "v3 array",
			files:     map[string]string{"n/zarr.json": v3ArrayDoc("float32", "[2]", "0", bytesLE)},
			open:      openWant{typ: "*zarr.Array[float32]"},
			openArray: openWant{typ: "*zarr.Array[float32]"},
		},
		{
			name:      "v3 group",
			files:     map[string]string{"n/zarr.json": v3GroupDoc},
			open:      openWant{typ: "*zarr.Group", attr: "v3"},
			openArray: openWant{msg: "not an array"},
		},
		{
			name:      "v2 array",
			files:     map[string]string{"n/.zarray": v2F4Array, "n/.zattrs": v2AttrsDoc},
			open:      openWant{typ: "*zarr.Array[float32]"},
			openArray: openWant{typ: "*zarr.Array[float32]"},
		},
		{
			name:      "v2 group",
			files:     map[string]string{"n/.zgroup": v2GroupDoc, "n/.zattrs": v2AttrsDoc},
			open:      openWant{typ: "*zarr.Group", attr: "v2"},
			openArray: openWant{msg: "not an array"},
		},
		{
			name:      "v3 wins over v2",
			files:     map[string]string{"n/zarr.json": v3GroupDoc, "n/.zarray": v2F4Array},
			open:      openWant{typ: "*zarr.Group", attr: "v3"},
			openArray: openWant{msg: "not an array"},
		},
		{
			name:      "missing",
			files:     map[string]string{"other/zarr.json": v3GroupDoc},
			open:      openWant{msg: "no v2 or v3 metadata found at"},
			openArray: openWant{msg: "no v2 or v3 metadata found at"},
		},
		{
			name:      "corrupt zarr.json",
			files:     map[string]string{"n/zarr.json": corruptJSON},
			open:      openWant{is: corrupt},
			openArray: openWant{is: corrupt},
		},
		{
			name:      "corrupt .zarray",
			files:     map[string]string{"n/.zarray": `{"zarr_format": `},
			open:      openWant{is: corrupt},
			openArray: openWant{is: corrupt},
		},
		{
			name:      "corrupt .zgroup",
			files:     map[string]string{"n/.zgroup": `{"zarr_format": `},
			open:      openWant{is: corrupt},
			openArray: openWant{is: corrupt},
		},
		{
			name:      "corrupt .zattrs",
			files:     map[string]string{"n/.zarray": v2F4Array, "n/.zattrs": `{`},
			open:      openWant{msg: "parse .zattrs"},
			openArray: openWant{msg: "parse .zattrs"},
		},
		{
			name:      "unknown node_type",
			files:     map[string]string{"n/zarr.json": `{"zarr_format": 3, "node_type": "blob"}`},
			open:      openWant{is: corrupt, msg: "blob"},
			openArray: openWant{is: corrupt, msg: "blob"},
		},
		{
			name:      "zero chunk dim",
			files:     map[string]string{"n/zarr.json": v3ArrayDoc("float32", "[0]", "0", bytesLE)},
			open:      openWant{is: corrupt},
			openArray: openWant{is: corrupt},
		},
		{
			name:      "unknown codec",
			files:     map[string]string{"n/zarr.json": v3ArrayDoc("float32", "[2]", "0", `[{"name": "not_a_codec"}]`)},
			open:      openWant{is: []error{zarr.ErrUnknownCodec}},
			openArray: openWant{is: []error{zarr.ErrUnknownCodec}},
		},
		{
			name:      "unsupported dtype",
			files:     map[string]string{"n/zarr.json": v3ArrayDoc("r16", "[2]", "0", bytesLE)},
			open:      openWant{is: []error{zarr.ErrUnsupportedDtype}},
			openArray: openWant{is: []error{zarr.ErrUnsupportedDtype}},
		},
		{
			name:      "bad fill_value",
			files:     map[string]string{"n/zarr.json": v3ArrayDoc("float32", "[2]", `"bogus"`, bytesLE)},
			open:      openWant{msg: "parse fill_value"},
			openArray: openWant{msg: "parse fill_value"},
		},
		{
			name:      "dtype mismatch",
			files:     map[string]string{"n/zarr.json": v3ArrayDoc("int32", "[2]", "0", bytesLE)},
			open:      openWant{typ: "*zarr.Array[int32]"},
			openArray: openWant{msg: "dtype mismatch"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			s := mem.New()
			for k, v := range tc.files {
				if err := s.Set(ctx, k, []byte(v)); err != nil {
					t.Fatal(err)
				}
			}
			n, err := zarr.Open(ctx, s, "n")
			checkOpen(t, "Open", n, err, tc.open)
			a, err := zarr.OpenArray[float32](ctx, s, "n")
			if err != nil && a != nil {
				t.Fatalf("OpenArray: non-nil array alongside error %v", err)
			}
			var node any = a
			if a == nil {
				node = nil
			}
			checkOpen(t, "OpenArray", node, err, tc.openArray)
		})
	}
}

// TestOpenV2ArrayAttrs checks that a v2 array's .zattrs sidecar is exposed
// through Array.Attrs (via both OpenArray and Open), and that a v2 array
// has no dimension names.
func TestOpenV2ArrayAttrs(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	for k, v := range map[string]string{"n/.zarray": v2F4Array, "n/.zattrs": v2AttrsDoc} {
		if err := s.Set(ctx, k, []byte(v)); err != nil {
			t.Fatal(err)
		}
	}
	want := map[string]any{"k": "v2"}
	a, err := zarr.OpenArray[float32](ctx, s, "n")
	if err != nil {
		t.Fatalf("OpenArray: %v", err)
	}
	if got := a.Attrs(); !reflect.DeepEqual(got, want) {
		t.Errorf("OpenArray: Attrs() = %#v, want %#v", got, want)
	}
	if got := a.DimensionNames(); got != nil {
		t.Errorf("OpenArray: DimensionNames() = %v, want nil", got)
	}
	n, err := zarr.Open(ctx, s, "n")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if got := n.(*zarr.Array[float32]).Attrs(); !reflect.DeepEqual(got, want) {
		t.Errorf("Open: Attrs() = %#v, want %#v", got, want)
	}
}
