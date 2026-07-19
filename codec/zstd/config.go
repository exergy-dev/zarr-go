// Shared configuration parsing for the zstd codec. Compiles regardless of
// which build-tag-selected Codec struct (zstd.go's pure-Go variant or
// cgo_zstd.go's cgo variant) is active, since both declare identically named
// Level/Checksum fields.
package zstd

import (
	"encoding/json"
	"fmt"
)

type configJSON struct {
	Level    *int  `json:"level,omitempty"`
	Checksum *bool `json:"checksum,omitempty"`
}

// Name returns the codec's v3 spec name.
func (c *Codec) Name() string { return "zstd" }

// Configure parses the v3 codec configuration object.
//
// Accepted shape: {"level": <int>, "checksum": <bool>}. Both fields are
// optional. "level" defaults to 0 (library default). "checksum" defaults to
// false.
func (c *Codec) Configure(raw json.RawMessage) error {
	c.Level = 0
	c.Checksum = false
	if len(raw) == 0 {
		return nil
	}
	var cfg configJSON
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return fmt.Errorf("zstd codec: invalid configuration: %w", err)
	}
	if cfg.Level != nil {
		c.Level = *cfg.Level
	}
	if cfg.Checksum != nil {
		c.Checksum = *cfg.Checksum
	}
	return nil
}
