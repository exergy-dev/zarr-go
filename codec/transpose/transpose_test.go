package transpose

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"testing"

	"github.com/exergy-dev/zarr-go/codec"
)

// newCodec returns a configured transpose codec or fails the test.
func newCodec(t *testing.T, order []int) *Codec {
	t.Helper()
	raw, err := json.Marshal(map[string]any{"order": order})
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	c, err := New(raw)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c
}

func TestName(t *testing.T) {
	c := &Codec{}
	if got := c.Name(); got != "transpose" {
		t.Fatalf("Name() = %q, want %q", got, "transpose")
	}
}

func TestRegistered(t *testing.T) {
	f := codec.Lookup("transpose")
	if f == nil {
		t.Fatal("transpose codec not registered")
	}
	c, err := f(json.RawMessage(`{"order":[1,0]}`))
	if err != nil {
		t.Fatalf("factory: %v", err)
	}
	if _, ok := c.(codec.ArrayToArrayCodec); !ok {
		t.Fatal("registered transpose codec does not implement ArrayToArrayCodec")
	}
}

func TestEncodeIdentity2D(t *testing.T) {
	c := newCodec(t, []int{0, 1})
	in := codec.ArrayChunk{
		Shape: []int64{2, 3},
		Dtype: "uint8",
		Data:  []byte{1, 2, 3, 4, 5, 6},
	}
	var out codec.ArrayChunk
	if err := c.EncodeAA(in, &out); err != nil {
		t.Fatalf("EncodeAA: %v", err)
	}
	if !bytes.Equal(out.Data, in.Data) {
		t.Fatalf("identity transpose changed data: got %v want %v", out.Data, in.Data)
	}
	if out.Shape[0] != 2 || out.Shape[1] != 3 {
		t.Fatalf("identity transpose changed shape: %v", out.Shape)
	}
	if out.Dtype != "uint8" {
		t.Fatalf("Dtype not propagated: %q", out.Dtype)
	}
}

func TestEncodeSwap2D(t *testing.T) {
	// 2x3 input:
	//   1 2 3
	//   4 5 6
	// Transpose -> 3x2:
	//   1 4
	//   2 5
	//   3 6
	c := newCodec(t, []int{1, 0})
	in := codec.ArrayChunk{
		Shape: []int64{2, 3},
		Dtype: "uint8",
		Data:  []byte{1, 2, 3, 4, 5, 6},
	}
	var out codec.ArrayChunk
	if err := c.EncodeAA(in, &out); err != nil {
		t.Fatalf("EncodeAA: %v", err)
	}
	want := []byte{1, 4, 2, 5, 3, 6}
	if !bytes.Equal(out.Data, want) {
		t.Fatalf("Encode [1,0]: got %v want %v", out.Data, want)
	}
	if out.Shape[0] != 3 || out.Shape[1] != 2 {
		t.Fatalf("Encode [1,0] shape: got %v want [3 2]", out.Shape)
	}

	// Decode round-trip back to 2x3.
	var back codec.ArrayChunk
	if err := c.DecodeAA(out, &back); err != nil {
		t.Fatalf("DecodeAA: %v", err)
	}
	if !bytes.Equal(back.Data, in.Data) {
		t.Fatalf("DecodeAA round-trip: got %v want %v", back.Data, in.Data)
	}
	if back.Shape[0] != 2 || back.Shape[1] != 3 {
		t.Fatalf("DecodeAA shape: %v", back.Shape)
	}
}

func TestEncodeRotation3D(t *testing.T) {
	// Input shape (2,3,4) with int32 elements [0..23].
	const D0, D1, D2 = 2, 3, 4
	const elem = 4
	in := codec.ArrayChunk{
		Shape: []int64{D0, D1, D2},
		Dtype: "int32",
		Data:  make([]byte, D0*D1*D2*elem),
	}
	for i := int32(0); i < D0*D1*D2; i++ {
		binary.LittleEndian.PutUint32(in.Data[int(i)*elem:], uint32(i))
	}

	c := newCodec(t, []int{2, 0, 1})
	var enc codec.ArrayChunk
	if err := c.EncodeAA(in, &enc); err != nil {
		t.Fatalf("EncodeAA: %v", err)
	}
	wantShape := []int64{D2, D0, D1}
	if enc.Shape[0] != wantShape[0] || enc.Shape[1] != wantShape[1] || enc.Shape[2] != wantShape[2] {
		t.Fatalf("Encode shape: got %v want %v", enc.Shape, wantShape)
	}

	// Spot-check: out[k,i,j] == in[i,j,k].
	for i := int64(0); i < D0; i++ {
		for j := int64(0); j < D1; j++ {
			for k := int64(0); k < D2; k++ {
				inIdx := i*D1*D2 + j*D2 + k
				outIdx := k*D0*D1 + i*D1 + j
				inV := binary.LittleEndian.Uint32(in.Data[inIdx*elem:])
				outV := binary.LittleEndian.Uint32(enc.Data[outIdx*elem:])
				if inV != outV {
					t.Fatalf("out[%d,%d,%d]=%d, in[%d,%d,%d]=%d", k, i, j, outV, i, j, k, inV)
				}
			}
		}
	}

	// Round-trip: DecodeAA should invert and recover original bytes/shape.
	var dec codec.ArrayChunk
	if err := c.DecodeAA(enc, &dec); err != nil {
		t.Fatalf("DecodeAA: %v", err)
	}
	if dec.Shape[0] != D0 || dec.Shape[1] != D1 || dec.Shape[2] != D2 {
		t.Fatalf("DecodeAA shape: got %v want [%d %d %d]", dec.Shape, D0, D1, D2)
	}
	if !bytes.Equal(dec.Data, in.Data) {
		t.Fatal("DecodeAA did not recover original data")
	}
}

func TestRoundTripFloat64(t *testing.T) {
	// Verify dtype with 8-byte elements works.
	const D0, D1 = 4, 5
	in := codec.ArrayChunk{
		Shape: []int64{D0, D1},
		Dtype: "float64",
		Data:  make([]byte, D0*D1*8),
	}
	for i := 0; i < D0*D1; i++ {
		binary.LittleEndian.PutUint64(in.Data[i*8:], math.Float64bits(float64(i)*0.5))
	}
	c := newCodec(t, []int{1, 0})
	var enc codec.ArrayChunk
	if err := c.EncodeAA(in, &enc); err != nil {
		t.Fatalf("EncodeAA: %v", err)
	}
	if enc.Shape[0] != D1 || enc.Shape[1] != D0 {
		t.Fatalf("shape: %v", enc.Shape)
	}
	var dec codec.ArrayChunk
	if err := c.DecodeAA(enc, &dec); err != nil {
		t.Fatalf("DecodeAA: %v", err)
	}
	if !bytes.Equal(dec.Data, in.Data) {
		t.Fatal("float64 round-trip mismatch")
	}
}

func TestRoundTripInt16(t *testing.T) {
	// Verify dtype with 2-byte elements works.
	const D0, D1, D2 = 2, 2, 3
	in := codec.ArrayChunk{
		Shape: []int64{D0, D1, D2},
		Dtype: "int16",
		Data:  make([]byte, D0*D1*D2*2),
	}
	for i := 0; i < D0*D1*D2; i++ {
		binary.LittleEndian.PutUint16(in.Data[i*2:], uint16(i+100))
	}
	c := newCodec(t, []int{2, 1, 0})
	var enc codec.ArrayChunk
	if err := c.EncodeAA(in, &enc); err != nil {
		t.Fatalf("EncodeAA: %v", err)
	}
	if enc.Shape[0] != D2 || enc.Shape[1] != D1 || enc.Shape[2] != D0 {
		t.Fatalf("shape: %v", enc.Shape)
	}
	var dec codec.ArrayChunk
	if err := c.DecodeAA(enc, &dec); err != nil {
		t.Fatalf("DecodeAA: %v", err)
	}
	if !bytes.Equal(dec.Data, in.Data) {
		t.Fatal("int16 round-trip mismatch")
	}
}

func TestReuseOutputBuffer(t *testing.T) {
	c := newCodec(t, []int{1, 0})
	in := codec.ArrayChunk{
		Shape: []int64{2, 3},
		Dtype: "uint8",
		Data:  []byte{1, 2, 3, 4, 5, 6},
	}
	pre := make([]byte, 100)
	out := codec.ArrayChunk{Data: pre}
	if err := c.EncodeAA(in, &out); err != nil {
		t.Fatalf("EncodeAA: %v", err)
	}
	if &out.Data[0] != &pre[0] {
		t.Fatal("expected output buffer to be reused when capacity is sufficient")
	}
	if len(out.Data) != 6 {
		t.Fatalf("len(out.Data)=%d want 6", len(out.Data))
	}
}

func TestNewRejectsBadOrders(t *testing.T) {
	cases := []struct {
		name string
		raw  string
	}{
		{"missing-order", `{}`},
		{"empty-order", `{"order": []}`},
		{"duplicate", `{"order": [0, 0]}`},
		{"out-of-range-high", `{"order": [0, 2]}`},
		{"out-of-range-negative", `{"order": [-1, 0]}`},
		{"missing-index", `{"order": [0, 0, 2]}`},
		{"not-json-object", `42`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, err := New(json.RawMessage(tc.raw))
			if err == nil {
				t.Fatalf("expected error for %s, got nil", tc.name)
			}
			if c != nil {
				t.Fatalf("New returned a codec alongside the error")
			}
		})
	}
}

func TestNewAcceptsValid(t *testing.T) {
	cases := [][]int{
		{0},
		{0, 1},
		{1, 0},
		{2, 0, 1},
		{1, 2, 0},
		{3, 1, 0, 2},
	}
	for _, order := range cases {
		raw, _ := json.Marshal(map[string]any{"order": order})
		if _, err := New(raw); err != nil {
			t.Fatalf("New(%v): %v", order, err)
		}
	}
}

func TestEncodeMismatchedRank(t *testing.T) {
	c := newCodec(t, []int{1, 0})
	in := codec.ArrayChunk{Shape: []int64{2, 3, 4}, Dtype: "uint8", Data: make([]byte, 24)}
	var out codec.ArrayChunk
	if err := c.EncodeAA(in, &out); err == nil {
		t.Fatal("expected error for rank mismatch")
	}
}

func TestEncodeUnsupportedDtype(t *testing.T) {
	c := newCodec(t, []int{1, 0})
	in := codec.ArrayChunk{Shape: []int64{2, 3}, Dtype: "string", Data: make([]byte, 6)}
	var out codec.ArrayChunk
	if err := c.EncodeAA(in, &out); err == nil {
		t.Fatal("expected error for unsupported dtype")
	}
}

func TestEncodeBadDataLength(t *testing.T) {
	c := newCodec(t, []int{1, 0})
	in := codec.ArrayChunk{Shape: []int64{2, 3}, Dtype: "int32", Data: make([]byte, 5)}
	var out codec.ArrayChunk
	if err := c.EncodeAA(in, &out); err == nil {
		t.Fatal("expected error for data length mismatch")
	}
}
