package sharding

import (
	"encoding/json"

	"github.com/exergy-dev/zarr-go/codec"
)

// Spec returns the metadata entry for a sharding_indexed codec, for
// zarr.WithCodecs. Each chunk of the array becomes a shard holding inner
// chunks of shape chunkShape, encoded with codecs; as in zarr.WithCodecs, a
// little-endian bytes codec is prepended when codecs has no array→bytes
// codec. The shard index uses the default codecs (bytes + crc32c) and is
// stored at the end of the shard.
//
// codecs must name registered codecs: an unknown codec, or one that rejects
// its configuration, is reported when the array is created.
func Spec(chunkShape []int64, codecs ...codec.Spec) codec.Spec {
	inner, err := codec.EnsureArrayToBytes(codecs)
	if err != nil {
		// Leave the list alone; building the pipeline reports the error.
		inner = codecs
	}
	if inner == nil {
		inner = []codec.Spec{}
	}
	cfg := configJSON{ChunkShape: chunkShape, Codecs: inner}
	if cfg.ChunkShape == nil {
		cfg.ChunkShape = []int64{}
	}
	raw, _ := json.Marshal(cfg) // ints and RawMessages cannot fail to marshal
	return codec.Spec{Name: "sharding_indexed", Configuration: raw}
}
