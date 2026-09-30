//go:build !cgo || !zarrcgo

package blosc

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/exergy-dev/zarr-go/codec"
)

func TestStubNewReturnsRequiresCGO(t *testing.T) {
	c, err := New(json.RawMessage(`{"cname":"lz4","clevel":5}`))
	if !errors.Is(err, codec.ErrRequiresCGO) {
		t.Fatalf("New err = %v, want codec.ErrRequiresCGO", err)
	}
	if c != nil {
		t.Fatalf("New returned a codec alongside the error: %v", c)
	}
}

func TestStubEncodeDecodeReturnsRequiresCGO(t *testing.T) {
	c := &Codec{}
	if _, err := c.EncodeBB([]byte("x")); !errors.Is(err, codec.ErrRequiresCGO) {
		t.Fatalf("EncodeBB err = %v, want codec.ErrRequiresCGO", err)
	}
	if _, err := c.DecodeBB([]byte("x")); !errors.Is(err, codec.ErrRequiresCGO) {
		t.Fatalf("DecodeBB err = %v, want codec.ErrRequiresCGO", err)
	}
}

func TestStubName(t *testing.T) {
	c := &Codec{}
	if c.Name() != "blosc" {
		t.Fatalf("Name() = %q, want %q", c.Name(), "blosc")
	}
}

// TestStubRegistration checks blosc is registered in the default build and
// that its factory reports ErrRequiresCGO (with no typed-nil codec), which
// BuildPipeline surfaces at array open.
func TestStubRegistration(t *testing.T) {
	f := codec.Lookup("blosc")
	if f == nil {
		t.Fatal("blosc codec not registered in default build")
	}
	inst, err := f(nil)
	if !errors.Is(err, codec.ErrRequiresCGO) {
		t.Fatalf("factory err = %v, want codec.ErrRequiresCGO", err)
	}
	if inst != nil {
		t.Fatalf("factory returned non-nil codec %#v alongside the error", inst)
	}
	// The factory fails before BuildPipeline checks codec ordering, so a
	// lone blosc spec is enough (and needs no other codec registered).
	_, err = codec.BuildPipeline([]codec.Spec{{Name: "blosc"}})
	if !errors.Is(err, codec.ErrRequiresCGO) {
		t.Fatalf("BuildPipeline err = %v, want codec.ErrRequiresCGO", err)
	}
}
