package zarr_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/exergy-dev/zarr-go/store/fs"
)

type v2ManifestEntry struct {
	Name       string  `json:"name"`
	Dtype      string  `json:"dtype"`
	Shape      []int64 `json:"shape"`
	Chunks     []int64 `json:"chunks"`
	Compressor *string `json:"compressor"`
	Separator  string  `json:"dimension_separator"`
}

// TestConformanceV2 reads Zarr v2 fixtures (produced by testdata/gen/generate_v2.py)
// and verifies byte-exact decoding against the paired expected.npy.
//
// Skipped when the v2 corpus isn't present; CI generates it before running.
func TestConformanceV2(t *testing.T) {
	corpusDir := "testdata/corpus_v2"
	manifestPath := filepath.Join(corpusDir, "manifest.json")
	raw, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Skipf("v2 corpus not present (%v); run testdata/gen/generate_v2.py", err)
	}
	var manifest []v2ManifestEntry
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
			if err := readAndCompareV2(t, s, entry, expected); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func readAndCompareV2(t *testing.T, s *fs.Store, entry v2ManifestEntry, expected []byte) error {
	t.Helper()
	// Reuse the v3 conformance harness; the manifest entry shape is identical.
	return readAndCompare(t, s, manifestEntry{
		Name:   entry.Name,
		Dtype:  entry.Dtype,
		Shape:  entry.Shape,
		Chunks: entry.Chunks,
	}, expected)
}
