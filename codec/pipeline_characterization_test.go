package codec_test

// Characterization tests pinning the behavior every registered codec must
// keep across changes to how codecs are constructed:
//
//   - a built Pipeline is shared by all of an Array's reader goroutines, so
//     each codec's Encode/Decode must be safe for concurrent use (run these
//     under -race);
//   - a bad configuration is rejected by BuildPipeline itself, with an error
//     that names the offending codec.

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"strings"
	"sync"
	"testing"

	"github.com/exergy-dev/zarr-go/codec"
	_ "github.com/exergy-dev/zarr-go/codec/blosc"
	_ "github.com/exergy-dev/zarr-go/codec/bytes"
	_ "github.com/exergy-dev/zarr-go/codec/crc32c"
	_ "github.com/exergy-dev/zarr-go/codec/gzip"
	_ "github.com/exergy-dev/zarr-go/codec/sharding"
	_ "github.com/exergy-dev/zarr-go/codec/transpose"
	_ "github.com/exergy-dev/zarr-go/codec/zlib"
	_ "github.com/exergy-dev/zarr-go/codec/zstd"
)

func spec(name, cfg string) codec.Spec {
	s := codec.Spec{Name: name}
	if cfg != "" {
		s.Configuration = json.RawMessage(cfg)
	}
	return s
}

var bytesLE = spec("bytes", `{"endian":"little"}`)

// concurrentPipelines covers every registered codec at least once.
var concurrentPipelines = []struct {
	name  string
	specs []codec.Spec
}{
	{"bytes-little", []codec.Spec{bytesLE}},
	{"bytes-big", []codec.Spec{spec("bytes", `{"endian":"big"}`)}},
	{"transpose", []codec.Spec{spec("transpose", `{"order":[1,0]}`), bytesLE}},
	{"crc32c", []codec.Spec{bytesLE, spec("crc32c", "")}},
	{"gzip", []codec.Spec{bytesLE, spec("gzip", `{"level":5}`)}},
	{"zlib", []codec.Spec{bytesLE, spec("zlib", `{"level":3}`)}},
	{"zstd", []codec.Spec{bytesLE, spec("zstd", `{"level":3,"checksum":true}`)}},
	{"blosc", []codec.Spec{bytesLE, spec("blosc", `{"cname":"lz4","clevel":5,"shuffle":"shuffle","typesize":4}`)}},
	{"sharding-end", []codec.Spec{spec("sharding_indexed",
		`{"chunk_shape":[4,8],"codecs":[{"name":"bytes","configuration":{"endian":"little"}},{"name":"gzip","configuration":{"level":1}}]}`)}},
	{"sharding-start", []codec.Spec{spec("sharding_indexed",
		`{"chunk_shape":[8,4],"index_location":"start","codecs":[{"name":"bytes","configuration":{"endian":"big"}},{"name":"zstd"}],"index_codecs":[{"name":"bytes"},{"name":"crc32c"}]}`)}},
}

// TestPipelineConcurrentRoundTrip runs one shared Pipeline's Encode then
// Decode from many goroutines at once, each on distinct data.
func TestPipelineConcurrentRoundTrip(t *testing.T) {
	const (
		goroutines = 16
		iterations = 50
	)
	shape := []int64{16, 16}
	const n = 16 * 16

	for _, tc := range concurrentPipelines {
		t.Run(tc.name, func(t *testing.T) {
			p, err := codec.BuildPipeline(tc.specs)
			if errors.Is(err, codec.ErrRequiresCGO) {
				t.Skip("codec requires a cgo build")
			}
			if err != nil {
				t.Fatalf("BuildPipeline: %v", err)
			}
			var wg sync.WaitGroup
			errs := make(chan error, goroutines)
			for g := range goroutines {
				wg.Go(func() {
					rng := rand.New(rand.NewPCG(uint64(g), 0x5eed))
					for it := range iterations {
						data := make([]byte, n*4)
						for i := range n {
							// Nonzero, goroutine- and iteration-specific
							// values, compressible enough to exercise the
							// codecs' real paths.
							v := uint32(g+1)<<24 | uint32(it)<<12 | uint32(rng.IntN(64)+1)
							binary.LittleEndian.PutUint32(data[i*4:], v)
						}
						in := codec.ArrayChunk{Shape: shape, Dtype: "int32", Data: data}
						enc, err := p.Encode(in)
						if err != nil {
							errs <- fmt.Errorf("goroutine %d iter %d: Encode: %w", g, it, err)
							return
						}
						out := codec.ArrayChunk{Shape: shape, Dtype: "int32"}
						if err := p.Decode(enc, &out); err != nil {
							errs <- fmt.Errorf("goroutine %d iter %d: Decode: %w", g, it, err)
							return
						}
						if !bytes.Equal(out.Data, data) {
							errs <- fmt.Errorf("goroutine %d iter %d: round trip mismatch", g, it)
							return
						}
					}
				})
			}
			wg.Wait()
			close(errs)
			for err := range errs {
				t.Error(err)
			}
		})
	}
}

// TestBuildPipelineConfigErrors asserts BuildPipeline rejects each bad
// codec configuration with an error naming the codec.
func TestBuildPipelineConfigErrors(t *testing.T) {
	aa := func(s codec.Spec) []codec.Spec { return []codec.Spec{s, bytesLE} }
	ab := func(s codec.Spec) []codec.Spec { return []codec.Spec{s} }
	bb := func(s codec.Spec) []codec.Spec { return []codec.Spec{bytesLE, s} }
	shard := func(cfg string) []codec.Spec { return ab(spec("sharding_indexed", cfg)) }

	cases := []struct {
		name     string
		codec    string // must appear in the error
		pipeline []codec.Spec
	}{
		{"bytes/bad-endian", "bytes", ab(spec("bytes", `{"endian":"middle"}`))},
		{"bytes/endian-type", "bytes", ab(spec("bytes", `{"endian":1}`))},
		{"bytes/malformed", "bytes", ab(spec("bytes", `{`))},

		{"transpose/missing-order", "transpose", aa(spec("transpose", `{}`))},
		{"transpose/no-config", "transpose", aa(spec("transpose", ""))},
		{"transpose/empty-order", "transpose", aa(spec("transpose", `{"order":[]}`))},
		{"transpose/duplicate", "transpose", aa(spec("transpose", `{"order":[0,0]}`))},
		{"transpose/out-of-range", "transpose", aa(spec("transpose", `{"order":[0,2]}`))},

		{"crc32c/not-object", "crc32c", bb(spec("crc32c", `[1,2]`))},

		{"gzip/level-high", "gzip", bb(spec("gzip", `{"level":10}`))},
		{"gzip/level-low", "gzip", bb(spec("gzip", `{"level":-2}`))},
		{"gzip/level-type", "gzip", bb(spec("gzip", `{"level":"max"}`))},

		{"zlib/level-high", "zlib", bb(spec("zlib", `{"level":10}`))},
		{"zlib/level-low", "zlib", bb(spec("zlib", `{"level":-5}`))},
		{"zlib/level-type", "zlib", bb(spec("zlib", `{"level":true}`))},

		{"zstd/level-type", "zstd", bb(spec("zstd", `{"level":"fast"}`))},
		{"zstd/checksum-type", "zstd", bb(spec("zstd", `{"checksum":"yes"}`))},
		{"zstd/malformed", "zstd", bb(spec("zstd", `{`))},

		// In the default (non-cgo) build every blosc configuration fails
		// with ErrRequiresCGO; with -tags zarrcgo these are real config
		// errors. Either way BuildPipeline must fail naming blosc.
		{"blosc/bad-shuffle", "blosc", bb(spec("blosc", `{"shuffle":"sideways"}`))},
		{"blosc/bad-shuffle-num", "blosc", bb(spec("blosc", `{"shuffle":7}`))},
		{"blosc/shuffle-type", "blosc", bb(spec("blosc", `{"shuffle":true}`))},
		{"blosc/clevel", "blosc", bb(spec("blosc", `{"clevel":15}`))},

		{"sharding/no-config", "sharding_indexed", ab(spec("sharding_indexed", ""))},
		{"sharding/missing-chunk-shape", "sharding_indexed",
			shard(`{"codecs":[{"name":"bytes"}]}`)},
		{"sharding/zero-chunk-dim", "sharding_indexed",
			shard(`{"chunk_shape":[4,0],"codecs":[{"name":"bytes"}]}`)},
		{"sharding/bad-index-location", "sharding_indexed",
			shard(`{"chunk_shape":[4],"index_location":"middle","codecs":[{"name":"bytes"}]}`)},
		{"sharding/missing-codecs", "sharding_indexed",
			shard(`{"chunk_shape":[4]}`)},
		{"sharding/bad-inner-codec", "sharding_indexed",
			shard(`{"chunk_shape":[4],"codecs":[{"name":"bytes"},{"name":"gzip","configuration":{"level":99}}]}`)},
		{"sharding/unknown-inner-codec", "sharding_indexed",
			shard(`{"chunk_shape":[4],"codecs":[{"name":"no_such_codec"}]}`)},
		{"sharding/aa-index-codec", "sharding_indexed",
			shard(`{"chunk_shape":[4],"codecs":[{"name":"bytes"}],"index_codecs":[{"name":"transpose","configuration":{"order":[0]}},{"name":"bytes"}]}`)},
		{"sharding/malformed", "sharding_indexed", shard(`{`)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := codec.BuildPipeline(tc.pipeline)
			if err == nil {
				t.Fatalf("BuildPipeline succeeded (%+v), want error", p)
			}
			if !strings.Contains(err.Error(), tc.codec) {
				t.Fatalf("error %q does not mention codec %q", err, tc.codec)
			}
		})
	}
}
