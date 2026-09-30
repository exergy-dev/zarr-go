package zarr_test

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/exergy-dev/zarr-go"
	"github.com/exergy-dev/zarr-go/store/fs"
)

type manifestEntry struct {
	Name   string   `json:"name"`
	Dtype  string   `json:"dtype"`
	Shape  []int64  `json:"shape"`
	Chunks []int64  `json:"chunks"`
	Codecs []string `json:"codecs"`
}

// TestConformance iterates the testdata/corpus that the Python generator
// (testdata/gen/generate.py) produces and verifies byte-exact decoding against
// the paired expected.npy file.
//
// The test is skipped when the corpus isn't present — local Python may not be
// new enough to run zarr 3 stable; CI installs the right deps and runs the
// generator before `go test`.
func TestConformance(t *testing.T) {
	corpusDir := "testdata/corpus"
	manifestPath := filepath.Join(corpusDir, "manifest.json")
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Skipf("corpus not present (%v); run testdata/gen/generate.py to create it", err)
	}
	var manifest []manifestEntry
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	for _, entry := range manifest {
		entry := entry
		t.Run(entry.Name, func(t *testing.T) {
			caseDir := filepath.Join(corpusDir, entry.Name)
			s, err := fs.New(filepath.Join(caseDir, "zarr"))
			if err != nil {
				t.Fatalf("open store: %v", err)
			}
			expected, err := readNPY(filepath.Join(caseDir, "expected.npy"))
			if err != nil {
				t.Fatalf("read expected.npy: %v", err)
			}
			if err := readAndCompare(t, s, entry, expected); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// readTyped opens the array at s's root as dtype T and reads sel into a
// freshly allocated slice.
func readTyped[T zarr.Numeric](ctx context.Context, s *fs.Store, sel zarr.Selection) ([]T, error) {
	a, err := zarr.OpenArray[T](ctx, s, "")
	if err != nil {
		return nil, fmt.Errorf("open: %w", err)
	}
	dst := make([]T, sel.NumElements())
	if err := a.Read(ctx, sel, dst); err != nil {
		return nil, fmt.Errorf("read: %w", err)
	}
	return dst, nil
}

func readAndCompare(t *testing.T, s *fs.Store, entry manifestEntry, expected []byte) error {
	t.Helper()
	ctx := context.Background()
	sel := zarr.All(entry.Shape)
	switch entry.Dtype {
	case "int8":
		dst, err := readTyped[int8](ctx, s, sel)
		if err != nil {
			return err
		}
		got := make([]byte, len(dst))
		for i, v := range dst {
			got[i] = byte(v)
		}
		return cmpBytes(got, expected)
	case "int16":
		dst, err := readTyped[int16](ctx, s, sel)
		if err != nil {
			return err
		}
		got := make([]byte, len(dst)*2)
		for i, v := range dst {
			binary.LittleEndian.PutUint16(got[i*2:], uint16(v))
		}
		return cmpBytes(got, expected)
	case "int32":
		dst, err := readTyped[int32](ctx, s, sel)
		if err != nil {
			return err
		}
		got := make([]byte, len(dst)*4)
		for i, v := range dst {
			binary.LittleEndian.PutUint32(got[i*4:], uint32(v))
		}
		return cmpBytes(got, expected)
	case "int64":
		dst, err := readTyped[int64](ctx, s, sel)
		if err != nil {
			return err
		}
		got := make([]byte, len(dst)*8)
		for i, v := range dst {
			binary.LittleEndian.PutUint64(got[i*8:], uint64(v))
		}
		return cmpBytes(got, expected)
	case "uint8":
		dst, err := readTyped[uint8](ctx, s, sel)
		if err != nil {
			return err
		}
		return cmpBytes(dst, expected)
	case "uint16":
		dst, err := readTyped[uint16](ctx, s, sel)
		if err != nil {
			return err
		}
		got := make([]byte, len(dst)*2)
		for i, v := range dst {
			binary.LittleEndian.PutUint16(got[i*2:], v)
		}
		return cmpBytes(got, expected)
	case "uint32":
		dst, err := readTyped[uint32](ctx, s, sel)
		if err != nil {
			return err
		}
		got := make([]byte, len(dst)*4)
		for i, v := range dst {
			binary.LittleEndian.PutUint32(got[i*4:], v)
		}
		return cmpBytes(got, expected)
	case "uint64":
		dst, err := readTyped[uint64](ctx, s, sel)
		if err != nil {
			return err
		}
		got := make([]byte, len(dst)*8)
		for i, v := range dst {
			binary.LittleEndian.PutUint64(got[i*8:], v)
		}
		return cmpBytes(got, expected)
	case "float16":
		dst, err := readTyped[zarr.Float16](ctx, s, sel)
		if err != nil {
			return err
		}
		got := make([]byte, len(dst)*2)
		for i, v := range dst {
			binary.LittleEndian.PutUint16(got[i*2:], uint16(v))
		}
		return cmpBytes(got, expected)
	case "float32":
		dst, err := readTyped[float32](ctx, s, sel)
		if err != nil {
			return err
		}
		got := make([]byte, len(dst)*4)
		for i, v := range dst {
			binary.LittleEndian.PutUint32(got[i*4:], math.Float32bits(v))
		}
		return cmpBytes(got, expected)
	case "float64":
		dst, err := readTyped[float64](ctx, s, sel)
		if err != nil {
			return err
		}
		got := make([]byte, len(dst)*8)
		for i, v := range dst {
			binary.LittleEndian.PutUint64(got[i*8:], math.Float64bits(v))
		}
		return cmpBytes(got, expected)
	case "complex64":
		dst, err := readTyped[complex64](ctx, s, sel)
		if err != nil {
			return err
		}
		got := make([]byte, len(dst)*8)
		for i, v := range dst {
			binary.LittleEndian.PutUint32(got[i*8:], math.Float32bits(real(v)))
			binary.LittleEndian.PutUint32(got[i*8+4:], math.Float32bits(imag(v)))
		}
		return cmpBytes(got, expected)
	case "complex128":
		dst, err := readTyped[complex128](ctx, s, sel)
		if err != nil {
			return err
		}
		got := make([]byte, len(dst)*16)
		for i, v := range dst {
			binary.LittleEndian.PutUint64(got[i*16:], math.Float64bits(real(v)))
			binary.LittleEndian.PutUint64(got[i*16+8:], math.Float64bits(imag(v)))
		}
		return cmpBytes(got, expected)
	case "bool":
		dst, err := readTyped[bool](ctx, s, sel)
		if err != nil {
			return err
		}
		got := make([]byte, len(dst))
		for i, v := range dst {
			if v {
				got[i] = 1
			}
		}
		return cmpBytes(got, expected)
	}
	return fmt.Errorf("dtype %s not yet wired in conformance harness", entry.Dtype)
}

func cmpBytes(got, want []byte) error {
	if !reflect.DeepEqual(got, want) {
		return fmt.Errorf("byte mismatch: got %d bytes, want %d bytes; first diff offset=%d",
			len(got), len(want), firstDiff(got, want))
	}
	return nil
}

func firstDiff(a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// readNPY loads a NumPy .npy file produced with allow_pickle=False and returns
// the raw element bytes in row-major order. Supports v1.0 and v2.0 headers.
func readNPY(path string) ([]byte, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 10 || string(raw[0:6]) != "\x93NUMPY" {
		return nil, fmt.Errorf("not a NPY file")
	}
	major := raw[6]
	var headerLen int
	var bodyStart int
	switch major {
	case 1:
		if len(raw) < 10 {
			return nil, fmt.Errorf("truncated NPY v1 header")
		}
		headerLen = int(binary.LittleEndian.Uint16(raw[8:10]))
		bodyStart = 10 + headerLen
	case 2, 3:
		if len(raw) < 12 {
			return nil, fmt.Errorf("truncated NPY v%d header", major)
		}
		headerLen = int(binary.LittleEndian.Uint32(raw[8:12]))
		bodyStart = 12 + headerLen
	default:
		return nil, fmt.Errorf("unsupported NPY major %d", major)
	}
	if bodyStart > len(raw) {
		return nil, fmt.Errorf("NPY header overruns file")
	}
	return raw[bodyStart:], nil
}
