package zarr_test

import (
	"context"
	"fmt"
	"log"
	"math"

	zarr "github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/codec/zstd"
	"github.com/exergy-dev/zarr-go/store/mem"
)

// Create a compressed array, write it, and read a window back.
func Example() {
	ctx := context.Background()
	st := mem.New() // or fs.New(dir) for a directory on disk

	shape, chunks := []int64{4, 6}, []int64{2, 3}
	a, err := zarr.CreateArray[float32](ctx, st, "temperature", shape, chunks,
		zarr.WithCodecs(zstd.Spec(3)),
		zarr.WithFillValue(float32(math.NaN())),
		zarr.WithDimensionNames([]string{"y", "x"}),
	)
	if err != nil {
		log.Fatal(err)
	}

	data := make([]float32, 4*6)
	for i := range data {
		data[i] = float32(i)
	}
	if err := a.Write(ctx, zarr.All(shape), data); err != nil {
		log.Fatal(err)
	}

	b, err := zarr.OpenArray[float32](ctx, st, "temperature")
	if err != nil {
		log.Fatal(err)
	}
	// Rows 1-2, columns 2-4.
	sel := zarr.Selection{{Start: 1, Stop: 3}, {Start: 2, Stop: 5}}
	out := make([]float32, sel.NumElements())
	if err := b.Read(ctx, sel, out); err != nil {
		log.Fatal(err)
	}
	fmt.Println(b.Shape(), b.Chunks(), b.Dtype(), b.DimensionNames())
	fmt.Println(out)
	// Output:
	// [4 6] [2 3] float32 [y x]
	// [8 9 10 14 15 16]
}

// Open an array whose dtype is not known at compile time.
func ExampleOpen() {
	ctx := context.Background()
	st := mem.New()
	if _, err := zarr.CreateArray[int16](ctx, st, "counts", []int64{3}, []int64{3}); err != nil {
		log.Fatal(err)
	}

	node, err := zarr.Open(ctx, st, "counts")
	if err != nil {
		log.Fatal(err)
	}
	switch n := node.(type) {
	case *zarr.Array[int16]:
		fmt.Println("int16 array of shape", n.Shape())
	case *zarr.Array[float32]:
		fmt.Println("float32 array of shape", n.Shape())
	case *zarr.Group:
		fmt.Println("group at", n.Path())
	}
	// Output:
	// int16 array of shape [3]
}

// Stream an array chunk by chunk instead of materializing the selection.
func ExampleArray_Stream() {
	ctx := context.Background()
	st := mem.New()
	shape, chunks := []int64{5}, []int64{2}
	a, err := zarr.CreateArray[int32](ctx, st, "", shape, chunks, zarr.WithFillValue(-1))
	if err != nil {
		log.Fatal(err)
	}
	// Only the first chunk is written; the others read as the fill value.
	if err := a.Write(ctx, zarr.Selection{{Start: 0, Stop: 2}}, []int32{10, 11}); err != nil {
		log.Fatal(err)
	}

	for c, err := range a.Stream(ctx, zarr.All(shape)) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(c.Coord, c.Shape, c.Data)
	}
	// Output:
	// [0] [2] [10 11]
	// [1] [2] [-1 -1]
	// [2] [1] [-1]
}

// Groups hold arrays and other groups; paths are slash-separated.
func ExampleCreateGroup() {
	ctx := context.Background()
	st := mem.New()
	g, err := zarr.CreateGroup(ctx, st, "experiment", zarr.WithAttributes(map[string]any{"run": 7}))
	if err != nil {
		log.Fatal(err)
	}
	a, err := zarr.CreateArray[uint8](ctx, st, g.Path()+"/mask", []int64{2, 2}, []int64{2, 2})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(g.Path(), g.Attrs()["run"], a.Path())
	// Output:
	// experiment 7 experiment/mask
}
