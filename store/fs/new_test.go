package fs

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestNewCreatesMissingRoot(t *testing.T) {
	root := filepath.Join(t.TempDir(), "nested", "out.zarr")
	s, err := New(root)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	info, err := os.Stat(root)
	if err != nil || !info.IsDir() {
		t.Fatalf("root not created: %v", err)
	}
	if err := s.Set(context.Background(), "zarr.json", []byte("{}")); err != nil {
		t.Fatalf("Set: %v", err)
	}
}

func TestNewRejectsFileRoot(t *testing.T) {
	f := filepath.Join(t.TempDir(), "file")
	if err := os.WriteFile(f, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := New(f); err == nil {
		t.Fatal("New(file) = nil, want error")
	}
}
