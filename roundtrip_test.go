package zarr_test

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/rkm/zarr-go"
	"github.com/rkm/zarr-go/codec"
	"github.com/rkm/zarr-go/store/fs"
)

// TestRoundTripPython writes arrays with zarr-go, then invokes zarr-python to
// read them back and compare against an .npy ground truth. The test is skipped
// when the venv from testdata/gen/ is not present.
func TestRoundTripPython(t *testing.T) {
	venvPython := "testdata/gen/.venv/bin/python"
	if _, err := os.Stat(venvPython); err != nil {
		t.Skipf("venv not present (%v); set up testdata/gen/.venv first", err)
	}
	verifier := "testdata/gen/verify_roundtrip.py"
	if _, err := os.Stat(verifier); err != nil {
		writeVerifier(t, verifier)
	}

	cases := []struct {
		name   string
		shape  []int64
		chunks []int64
		dtype  string
		codecs []codec.Spec
		fill   json.RawMessage
	}{
		{"f32_bytes", []int64{8, 8}, []int64{4, 4}, "float32", nil, nil},
		{"f64_gzip", []int64{8, 8}, []int64{4, 4}, "float64",
			[]codec.Spec{
				{Name: "bytes", Configuration: json.RawMessage(`{"endian":"little"}`)},
				{Name: "gzip", Configuration: json.RawMessage(`{"level":5}`)},
			}, nil},
		{"i32_zstd", []int64{4, 4, 4}, []int64{2, 2, 2}, "int32",
			[]codec.Spec{
				{Name: "bytes", Configuration: json.RawMessage(`{"endian":"little"}`)},
				{Name: "zstd", Configuration: json.RawMessage(`{"level":3,"checksum":false}`)},
			}, nil},
		{"f32_edge", []int64{10}, []int64{4}, "float32", nil, json.RawMessage("0")},
	}

	for _, c := range cases {
		c := c
		t.Run(c.name, func(t *testing.T) {
			tmp := t.TempDir()
			zarrDir := filepath.Join(tmp, "arr")
			npyPath := filepath.Join(tmp, "expected.npy")

			s, err := fs.New(tmp)
			if err != nil {
				t.Fatal(err)
			}
			ctx := context.Background()

			data, npyBody, err := makeData(c.dtype, c.shape)
			if err != nil {
				t.Fatal(err)
			}
			if err := writeNPY(npyPath, c.dtype, c.shape, npyBody); err != nil {
				t.Fatal(err)
			}

			opts := []zarr.CreateOption{}
			if len(c.codecs) > 0 {
				opts = append(opts, zarr.WithCodecs(c.codecs...))
			}
			if len(c.fill) > 0 {
				opts = append(opts, zarr.WithFillValue(c.fill))
			}
			if err := rtWriteArray(t, ctx, s, "arr", c.shape, c.chunks, c.dtype, data, opts); err != nil {
				t.Fatal(err)
			}

			out, err := exec.Command(venvPython, verifier, zarrDir, npyPath).CombinedOutput()
			if err != nil {
				t.Fatalf("python verify failed: %v\n%s", err, out)
			}
			t.Logf("%s", out)
		})
	}
}

func rtWriteArray(t *testing.T, ctx context.Context, s *fs.Store, path string, shape, chunks []int64,
	dtype string, data any, opts []zarr.CreateOption) error {
	t.Helper()
	switch d := data.(type) {
	case []float32:
		a, err := zarr.CreateArray[float32](ctx, s, path, shape, chunks, opts...)
		if err != nil {
			return err
		}
		return a.Write(ctx, zarr.All(shape), d)
	case []float64:
		a, err := zarr.CreateArray[float64](ctx, s, path, shape, chunks, opts...)
		if err != nil {
			return err
		}
		return a.Write(ctx, zarr.All(shape), d)
	case []int32:
		a, err := zarr.CreateArray[int32](ctx, s, path, shape, chunks, opts...)
		if err != nil {
			return err
		}
		return a.Write(ctx, zarr.All(shape), d)
	}
	t.Fatalf("unsupported dtype %s", dtype)
	return nil
}

func makeData(dtype string, shape []int64) (any, []byte, error) {
	n := int64(1)
	for _, d := range shape {
		n *= d
	}
	switch dtype {
	case "float32":
		v := make([]float32, n)
		b := make([]byte, n*4)
		for i := range v {
			v[i] = float32(i)*0.5 - 1
			binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v[i]))
		}
		return v, b, nil
	case "float64":
		v := make([]float64, n)
		b := make([]byte, n*8)
		for i := range v {
			v[i] = float64(i)*0.25 + 3.14
			binary.LittleEndian.PutUint64(b[i*8:], math.Float64bits(v[i]))
		}
		return v, b, nil
	case "int32":
		v := make([]int32, n)
		b := make([]byte, n*4)
		for i := range v {
			v[i] = int32(i*7 - 100)
			binary.LittleEndian.PutUint32(b[i*4:], uint32(v[i]))
		}
		return v, b, nil
	}
	return nil, nil, nil
}

// writeNPY writes a minimal NPY v1.0 file.
func writeNPY(path, dtype string, shape []int64, body []byte) error {
	npyDtype := map[string]string{
		"float32": "<f4",
		"float64": "<f8",
		"int32":   "<i4",
		"int16":   "<i2",
		"uint8":   "|u1",
	}[dtype]
	if npyDtype == "" {
		return os.ErrInvalid
	}
	shapeStr := "("
	for i, d := range shape {
		if i > 0 {
			shapeStr += ", "
		}
		shapeStr += itoaInt(d)
	}
	if len(shape) == 1 {
		shapeStr += ","
	}
	shapeStr += ")"
	header := "{'descr': '" + npyDtype + "', 'fortran_order': False, 'shape': " + shapeStr + ", }"
	// Pad header to a multiple of 64 (with trailing \n).
	totalHeader := 10 + len(header) + 1
	pad := (64 - totalHeader%64) % 64
	for i := 0; i < pad; i++ {
		header += " "
	}
	header += "\n"

	out := make([]byte, 0, 10+len(header)+len(body))
	out = append(out, '\x93', 'N', 'U', 'M', 'P', 'Y', 1, 0)
	out = append(out, byte(len(header)&0xff), byte((len(header)>>8)&0xff))
	out = append(out, header...)
	out = append(out, body...)
	return os.WriteFile(path, out, 0644)
}

func itoaInt(n int64) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

func writeVerifier(t *testing.T, path string) {
	t.Helper()
	script := `import sys, numpy as np
try:
    from zarr.storage import LocalStore
except ImportError:
    from zarr.store import LocalStore
import zarr
arr = zarr.open(LocalStore(sys.argv[1], mode='r'), mode='r')
data = arr[:]
expected = np.load(sys.argv[2])
assert data.shape == expected.shape, f"shape: {data.shape} != {expected.shape}"
assert np.array_equal(data, expected), "values differ"
print(f"OK shape={data.shape} dtype={data.dtype}")
`
	if err := os.WriteFile(path, []byte(script), 0644); err != nil {
		t.Fatal(err)
	}
}
