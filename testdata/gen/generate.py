"""Generate the Zarr v3 conformance corpus for go-zarr.

Each case writes:
  testdata/corpus/<name>/zarr/         -- a Zarr v3 store
  testdata/corpus/<name>/expected.npy  -- the ground-truth array

A top-level testdata/corpus/manifest.json lists every case with its
dtype, shape, chunks and codec pipeline so the Go test driver can
iterate them and verify byte-exact decoding.

Reproducible: all random data is drawn from np.random.default_rng(42).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from zarr.codecs import (
    BytesCodec,
    Crc32cCodec,
    GzipCodec,
    ShardingCodec,
    ZstdCodec,
)
try:
    from zarr.storage import LocalStore  # zarr 3 stable
except ImportError:
    from zarr.store import LocalStore    # zarr 3 alpha


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GEN_DIR = Path(__file__).resolve().parent
TESTDATA_DIR = GEN_DIR.parent
CORPUS_DIR = TESTDATA_DIR / "corpus"


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def make_data(rng: np.random.Generator, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    """Produce deterministic test data for the given dtype/shape."""
    np_dtype = np.dtype(dtype)
    if np_dtype.kind == "f":
        return rng.standard_normal(shape).astype(np_dtype)
    if np_dtype.kind == "i":
        info = np.iinfo(np_dtype)
        # Keep range modest so values are easy to read in test diagnostics.
        low = max(info.min, -10_000)
        high = min(info.max, 10_000)
        return rng.integers(low, high, size=shape, dtype=np_dtype)
    if np_dtype.kind == "u":
        info = np.iinfo(np_dtype)
        high = min(int(info.max) + 1, 256)
        return rng.integers(0, high, size=shape, dtype=np_dtype)
    if np_dtype.kind == "b":
        return rng.integers(0, 2, size=shape).astype(bool)
    raise ValueError(f"unsupported dtype: {dtype}")


# ---------------------------------------------------------------------------
# Codec pipeline construction
# ---------------------------------------------------------------------------

def build_codecs(spec: list[dict[str, Any]]) -> list[Any]:
    """Translate a manifest-friendly codec spec list into zarr codec objects."""
    codecs: list[Any] = []
    for item in spec:
        name = item["name"]
        if name == "bytes":
            codecs.append(BytesCodec())
        elif name == "gzip":
            codecs.append(GzipCodec(level=item.get("level", 5)))
        elif name == "zstd":
            codecs.append(ZstdCodec(level=item.get("level", 3)))
        elif name == "crc32c":
            codecs.append(Crc32cCodec())
        elif name == "sharding_indexed":
            inner = build_codecs(item["codecs"])
            codecs.append(
                ShardingCodec(
                    chunk_shape=tuple(item["chunk_shape"]),
                    codecs=inner,
                )
            )
        else:
            raise ValueError(f"unknown codec: {name}")
    return codecs


def codec_names(spec: list[dict[str, Any]]) -> list[str]:
    """Flatten a codec spec to just the codec names for the manifest."""
    out: list[str] = []
    for item in spec:
        if item["name"] == "sharding_indexed":
            inner = "+".join(codec_names(item["codecs"]))
            out.append(f"sharding_indexed({inner})")
        elif item["name"] in ("gzip", "zstd") and "level" in item:
            out.append(f"{item['name']}(level={item['level']})")
        else:
            out.append(item["name"])
    return out


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

CASES: list[dict[str, Any]] = [
    {
        "name": "f32_1d_bytes_only",
        "dtype": "float32",
        "shape": [128],
        "chunks": [16],
        "codec_spec": [{"name": "bytes"}],
    },
    {
        "name": "f64_2d_bytes_only",
        "dtype": "float64",
        "shape": [16, 16],
        "chunks": [4, 8],
        "codec_spec": [{"name": "bytes"}],
    },
    {
        "name": "i32_3d_bytes_only",
        "dtype": "int32",
        "shape": [4, 8, 6],
        "chunks": [2, 4, 3],
        "codec_spec": [{"name": "bytes"}],
    },
    {
        "name": "u8_2d_bytes_only",
        "dtype": "uint8",
        "shape": [10, 10],
        "chunks": [5, 5],
        "codec_spec": [{"name": "bytes"}],
    },
    {
        "name": "bool_1d_bytes_only",
        "dtype": "bool",
        "shape": [64],
        "chunks": [16],
        "codec_spec": [{"name": "bytes"}],
    },
    {
        "name": "f32_2d_gzip",
        "dtype": "float32",
        "shape": [32, 32],
        "chunks": [8, 8],
        "codec_spec": [{"name": "bytes"}, {"name": "gzip", "level": 5}],
    },
    {
        "name": "f64_2d_zstd",
        "dtype": "float64",
        "shape": [16, 16],
        "chunks": [4, 4],
        "codec_spec": [{"name": "bytes"}, {"name": "zstd", "level": 3}],
    },
    {
        "name": "i16_3d_gzip_crc32c",
        "dtype": "int16",
        "shape": [4, 4, 4],
        "chunks": [2, 2, 2],
        "codec_spec": [
            {"name": "bytes"},
            {"name": "gzip", "level": 5},
            {"name": "crc32c"},
        ],
    },
    {
        "name": "f32_2d_sharded",
        "dtype": "float32",
        "shape": [16, 16],
        "chunks": [8, 8],
        "codec_spec": [
            {
                "name": "sharding_indexed",
                "chunk_shape": [4, 4],
                "codecs": [{"name": "bytes"}],
            }
        ],
    },
    {
        "name": "f32_2d_sharded_gzip",
        "dtype": "float32",
        "shape": [16, 16],
        "chunks": [8, 8],
        "codec_spec": [
            {
                "name": "sharding_indexed",
                "chunk_shape": [4, 4],
                "codecs": [{"name": "bytes"}, {"name": "gzip", "level": 5}],
            }
        ],
    },
    # Edge-chunk coverage: shape not a multiple of chunk shape.
    {
        "name": "f32_edge",
        "dtype": "float32",
        "shape": [10],
        "chunks": [4],
        "codec_spec": [{"name": "bytes"}],
    },
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def generate_case(case: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    name = case["name"]
    dtype = case["dtype"]
    shape = tuple(case["shape"])
    chunks = tuple(case["chunks"])
    codec_spec = case["codec_spec"]

    case_dir = CORPUS_DIR / name
    zarr_dir = case_dir / "zarr"
    npy_path = case_dir / "expected.npy"
    case_dir.mkdir(parents=True, exist_ok=True)

    data = make_data(rng, dtype, shape)
    codecs = build_codecs(codec_spec)

    if hasattr(zarr, "create_array"):
        arr = zarr.create_array(
            store=LocalStore(str(zarr_dir)),
            shape=shape,
            chunks=chunks,
            dtype=np.dtype(dtype),
            codecs=codecs,
        )
    else:
        # zarr 3 alpha API: use zarr.create with mode='w'.
        arr = zarr.create(
            store=LocalStore(str(zarr_dir), mode="w"),
            shape=shape,
            chunks=chunks,
            dtype=np.dtype(dtype),
            codecs=codecs,
            zarr_format=3,
        )
    arr[:] = data

    np.save(npy_path, data, allow_pickle=False)

    return {
        "name": name,
        "dtype": dtype,
        "shape": list(shape),
        "chunks": list(chunks),
        "codecs": codec_names(codec_spec),
    }


def main() -> int:
    shutil.rmtree(CORPUS_DIR, ignore_errors=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    manifest: list[dict[str, Any]] = []
    for case in CASES:
        entry = generate_case(case, rng)
        manifest.append(entry)
        print(f"generated {entry['name']:<24} dtype={entry['dtype']:<7} "
              f"shape={entry['shape']} chunks={entry['chunks']} "
              f"codecs={entry['codecs']}")

    manifest_path = CORPUS_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print(f"wrote {len(manifest)} cases to {CORPUS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
