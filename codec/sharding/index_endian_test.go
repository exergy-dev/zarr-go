package sharding

import (
	"encoding/binary"
	"testing"
)

// TestIndexWireByteOrder pins the on-disk byte order of the shard index to
// the index_codecs' bytes-codec endian setting, independent of the host.
// The index table is handed to the bytes codec as native-order uint64s;
// building or reading it with a fixed byte order would only be right on
// hosts whose order happens to match (red on big-endian hosts before the
// switch to binary.NativeEndian).
func TestIndexWireByteOrder(t *testing.T) {
	entries := []uint64{0x0102030405060708, 0x1112131415161718}
	for _, tc := range []struct {
		endian string
		order  binary.ByteOrder
	}{
		{"little", binary.LittleEndian},
		{"big", binary.BigEndian},
	} {
		t.Run(tc.endian, func(t *testing.T) {
			c, err := New([]byte(`{"chunk_shape":[1],` +
				`"codecs":[{"name":"bytes","configuration":{"endian":"little"}}],` +
				`"index_codecs":[{"name":"bytes","configuration":{"endian":"` + tc.endian + `"}}]}`))
			if err != nil {
				t.Fatal(err)
			}
			wire, err := c.encodeIndexEntries(entries)
			if err != nil {
				t.Fatal(err)
			}
			for i, want := range entries {
				if got := tc.order.Uint64(wire[i*8:]); got != want {
					t.Errorf("wire entry %d = %#x, want %#x", i, got, want)
				}
			}
			back, err := c.decodeIndex(wire, 1)
			if err != nil {
				t.Fatal(err)
			}
			for i, want := range entries {
				if back[i] != want {
					t.Errorf("decoded entry %d = %#x, want %#x", i, back[i], want)
				}
			}
		})
	}
}
