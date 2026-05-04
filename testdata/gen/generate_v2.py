"""Generate Zarr v2 fixtures for go-zarr's v2 read path.

Each case writes:
  testdata/corpus_v2/<name>/zarr/         -- a Zarr v2 store
  testdata/corpus_v2/<name>/expected.npy  -- the ground-truth array

A top-level testdata/corpus_v2/manifest.json lists every case.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr  # zarr<3
from numcodecs import Blosc, GZip, Zstd, Zlib  # noqa: F401  (registered by import)

GEN_DIR = Path(__file__).resolve().parent
TESTDATA_DIR = GEN_DIR.parent
CORPUS_DIR = TESTDATA_DIR / "corpus_v2"


CASES = [
    {
        "name": "f32_1d_none",
        "dtype": "float32",
        "shape": [128],
        "chunks": [16],
        "compressor": None,
    },
    {
        "name": "f64_2d_none",
        "dtype": "float64",
        "shape": [16, 16],
        "chunks": [4, 8],
        "compressor": None,
    },
    {
        "name": "i32_3d_none",
        "dtype": "int32",
        "shape": [4, 8, 6],
        "chunks": [2, 4, 3],
        "compressor": None,
    },
    {
        "name": "u8_2d_none",
        "dtype": "uint8",
        "shape": [10, 10],
        "chunks": [5, 5],
        "compressor": None,
    },
    {
        "name": "f32_2d_gzip",
        "dtype": "float32",
        "shape": [32, 32],
        "chunks": [8, 8],
        "compressor": GZip(level=5),
    },
    {
        "name": "f64_2d_zstd",
        "dtype": "float64",
        "shape": [16, 16],
        "chunks": [4, 4],
        "compressor": Zstd(level=3),
    },
    {
        "name": "i16_2d_zlib",
        "dtype": "int16",
        "shape": [8, 8],
        "chunks": [4, 4],
        "compressor": Zlib(level=6),
    },
    {
        "name": "f32_edge",
        "dtype": "float32",
        "shape": [10],
        "chunks": [4],
        "compressor": None,
    },
    {
        "name": "f32_2d_dot_sep",
        "dtype": "float32",
        "shape": [8, 8],
        "chunks": [4, 4],
        "compressor": None,
        "dimension_separator": ".",
    },
]


def make_data(rng, dtype, shape):
    npd = np.dtype(dtype)
    if npd.kind == "f":
        return rng.standard_normal(shape).astype(npd)
    if npd.kind == "i":
        info = np.iinfo(npd)
        return rng.integers(max(info.min, -10000), min(info.max, 10000), size=shape, dtype=npd)
    if npd.kind == "u":
        info = np.iinfo(npd)
        return rng.integers(0, min(int(info.max) + 1, 256), size=shape, dtype=npd)
    raise ValueError(f"unsupported dtype: {dtype}")


def generate_case(case, rng):
    name = case["name"]
    case_dir = CORPUS_DIR / name
    zarr_dir = case_dir / "zarr"
    npy_path = case_dir / "expected.npy"
    case_dir.mkdir(parents=True, exist_ok=True)

    data = make_data(rng, case["dtype"], tuple(case["shape"]))
    sep = case.get("dimension_separator", "/")

    arr = zarr.create(
        store=zarr.DirectoryStore(str(zarr_dir)),
        shape=tuple(case["shape"]),
        chunks=tuple(case["chunks"]),
        dtype=np.dtype(case["dtype"]),
        compressor=case["compressor"],
        dimension_separator=sep,
    )
    arr[:] = data

    np.save(npy_path, data, allow_pickle=False)
    return {
        "name": name,
        "dtype": case["dtype"],
        "shape": list(case["shape"]),
        "chunks": list(case["chunks"]),
        "compressor": None if case["compressor"] is None else case["compressor"].codec_id,
        "dimension_separator": sep,
    }


def main():
    shutil.rmtree(CORPUS_DIR, ignore_errors=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    manifest = []
    for case in CASES:
        entry = generate_case(case, rng)
        manifest.append(entry)
        print(f"generated {entry['name']:<24} dtype={entry['dtype']:<7} "
              f"shape={entry['shape']} chunks={entry['chunks']} "
              f"compressor={entry['compressor']} sep={entry['dimension_separator']!r}")
    with (CORPUS_DIR / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"wrote {len(manifest)} cases to {CORPUS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
