package zarr_test

import (
	"context"
	"slices"
	"testing"

	"github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/store/mem"
)

// TestPathSpellingsRoundTrip checks that every spelling of a node path (with
// or without leading/trailing slashes) resolves to the same array: data
// written via one spelling must read back via every other. Before paths were
// normalized, metadata keys were slash-trimmed but chunk keys were not, so
// reading via "a/" or "/a" found the metadata, missed every chunk, and
// silently returned fill values.
func TestPathSpellingsRoundTrip(t *testing.T) {
	ctx := context.Background()
	classes := map[string][]string{
		"root":   {"", "/"},
		"a":      {"a", "a/", "/a", "/a/"},
		"nested": {"g/a", "/g/a", "g/a/", "/g/a/"},
	}
	want := []int32{7, 8, 9, 10}
	for name, spellings := range classes {
		for _, wpath := range spellings {
			for _, rpath := range spellings {
				t.Run(name+"/"+wpath+"->"+rpath, func(t *testing.T) {
					s := mem.New()
					a, err := zarr.CreateArray[int32](ctx, s, wpath, []int64{4}, []int64{2})
					if err != nil {
						t.Fatalf("CreateArray(%q): %v", wpath, err)
					}
					if err := a.Write(ctx, zarr.All([]int64{4}), want); err != nil {
						t.Fatalf("Write: %v", err)
					}

					b, err := zarr.OpenArray[int32](ctx, s, rpath)
					if err != nil {
						t.Fatalf("OpenArray(%q): %v", rpath, err)
					}
					got := make([]int32, 4)
					if err := b.Read(ctx, zarr.All([]int64{4}), got); err != nil {
						t.Fatalf("Read via OpenArray: %v", err)
					}
					if !slices.Equal(got, want) {
						t.Fatalf("OpenArray(%q) read %v, want %v", rpath, got, want)
					}

					n, err := zarr.Open(ctx, s, rpath)
					if err != nil {
						t.Fatalf("Open(%q): %v", rpath, err)
					}
					c, ok := n.(*zarr.Array[int32])
					if !ok {
						t.Fatalf("Open(%q) returned %T, want *Array[int32]", rpath, n)
					}
					got = make([]int32, 4)
					if err := c.Read(ctx, zarr.All([]int64{4}), got); err != nil {
						t.Fatalf("Read via Open: %v", err)
					}
					if !slices.Equal(got, want) {
						t.Fatalf("Open(%q) read %v, want %v", rpath, got, want)
					}
				})
			}
		}
	}
}

// TestRootArrayChunkKeys pins the exact key set written for an array at the
// store root: no chunk key may carry a leading "/".
func TestRootArrayChunkKeys(t *testing.T) {
	ctx := context.Background()
	cases := []struct {
		name string
		opts []zarr.CreateOption
		want []string
	}{
		{"default", nil, []string{"c/0", "c/1", "zarr.json"}},
		{"v2", []zarr.CreateOption{zarr.WithV2ChunkKeys()}, []string{"0", "1", "zarr.json"}},
	}
	for _, tc := range cases {
		for _, path := range []string{"", "/"} {
			t.Run(tc.name+"/"+path, func(t *testing.T) {
				s := mem.New()
				a, err := zarr.CreateArray[int32](ctx, s, path, []int64{4}, []int64{2}, tc.opts...)
				if err != nil {
					t.Fatalf("CreateArray: %v", err)
				}
				if err := a.Write(ctx, zarr.All([]int64{4}), []int32{1, 2, 3, 4}); err != nil {
					t.Fatalf("Write: %v", err)
				}
				var keys []string
				for k, err := range s.List(ctx, "") {
					if err != nil {
						t.Fatalf("List: %v", err)
					}
					keys = append(keys, k)
				}
				slices.Sort(keys)
				if !slices.Equal(keys, tc.want) {
					t.Fatalf("keys = %q, want %q", keys, tc.want)
				}
			})
		}
	}
}

// TestOverwriteDeletesChunksAcrossSpellings checks that a WithOverwrite
// re-create via a non-canonical spelling still clears the chunks written via
// the canonical one (deleteStaleChunks lists by path prefix).
func TestOverwriteDeletesChunksAcrossSpellings(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	a, err := zarr.CreateArray[int32](ctx, s, "a", []int64{4}, []int64{2})
	if err != nil {
		t.Fatalf("CreateArray: %v", err)
	}
	if err := a.Write(ctx, zarr.All([]int64{4}), []int32{1, 2, 3, 4}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if _, err := zarr.CreateArray[int32](ctx, s, "/a/", []int64{4}, []int64{2}, zarr.WithOverwrite()); err != nil {
		t.Fatalf("CreateArray WithOverwrite: %v", err)
	}
	for _, k := range []string{"a/c/0", "a/c/1"} {
		if ok, err := s.Exists(ctx, k); err != nil || ok {
			t.Fatalf("stale chunk %q still present after overwrite (ok=%v err=%v)", k, ok, err)
		}
	}
}

// TestGroupPathNormalized checks that CreateGroup / Group.CreateGroup store a
// canonical, slash-trimmed path.
func TestGroupPathNormalized(t *testing.T) {
	ctx := context.Background()
	s := mem.New()
	g, err := zarr.CreateGroup(ctx, s, "/g/")
	if err != nil {
		t.Fatalf("CreateGroup: %v", err)
	}
	if g.Path() != "g" {
		t.Fatalf("Path() = %q, want %q", g.Path(), "g")
	}
	sub, err := g.CreateGroup(ctx, "/h/")
	if err != nil {
		t.Fatalf("Group.CreateGroup: %v", err)
	}
	if sub.Path() != "g/h" {
		t.Fatalf("sub Path() = %q, want %q", sub.Path(), "g/h")
	}
	n, err := zarr.Open(ctx, s, "/g/h/")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if got := n.(*zarr.Group).Path(); got != "g/h" {
		t.Fatalf("Open Path() = %q, want %q", got, "g/h")
	}
}
