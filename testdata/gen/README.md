# zarr-go conformance test fixture generator

This directory contains the Python generator that produces the Zarr v3
conformance corpus consumed by the Go test suite.

## Running

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python generate.py
```

Output is written to `../corpus/` (gitignored). Each case becomes a
subdirectory containing a `zarr/` store and an `expected.npy` file with
the ground-truth element values. A top-level `manifest.json` enumerates
every case and its parameters so the Go test driver can iterate them.
