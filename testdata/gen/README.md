# zarr-go conformance test fixture generators

This directory contains two Python generators that produce the conformance
corpora consumed by the Go test suite:

- `generate.py` writes the Zarr **v3** corpus to `../corpus/`, using
  zarr-python 3.x (`requirements-v3.txt`).
- `generate_v2.py` writes the Zarr **v2** corpus to `../corpus_v2/`, using
  zarr-python 2.x (`requirements-v2.txt`).

The two scripts need incompatible major versions of the `zarr` package in
the same repo, so they cannot share one virtualenv/requirements file.

## Running (recommended: uv)

[uv](https://docs.astral.sh/uv/) runs each script in its own ephemeral,
reproducible environment with zero setup:

```bash
uv run --python 3.11 --with-requirements requirements-v3.txt generate.py
uv run --python 3.11 --with-requirements requirements-v2.txt generate_v2.py
```

`make fixtures` (from the repo root) runs both. zarr-python 3 requires
Python >= 3.11.

## Running (manual venv)

```bash
python3.11 -m venv .venv-v3
. .venv-v3/bin/activate
pip install -r requirements-v3.txt
python generate.py
deactivate

python3 -m venv .venv-v2
. .venv-v2/bin/activate
pip install -r requirements-v2.txt
python generate_v2.py
deactivate
```

## Output

Each case becomes a subdirectory containing a `zarr/` store and an
`expected.npy` file with the ground-truth element values (gitignored). A
top-level `manifest.json` in each corpus directory enumerates every case and
its parameters so the Go test driver (`TestConformance` / `TestConformanceV2`
in the repo root) can iterate them.
