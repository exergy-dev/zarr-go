//go:build !cgo || !zarrcgo

package blosc

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/rkm/zarr-go/codec"
)

func TestStubConfigureReturnsRequiresCGO(t *testing.T) {
	c := &Codec{}
	err := c.Configure(json.RawMessage(`{"cname":"lz4","clevel":5}`))
	if !errors.Is(err, codec.ErrRequiresCGO) {
		t.Fatalf("Configure err = %v, want codec.ErrRequiresCGO", err)
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

func TestStubRegistration(t *testing.T) {
	f := codec.Lookup("blosc")
	if f == nil {
		t.Fatal("blosc codec not registered in default build")
	}
	inst := f()
	if _, ok := inst.(codec.BytesToBytesCodec); !ok {
		t.Fatalf("registered blosc codec does not implement BytesToBytesCodec (got %T)", inst)
	}
}
