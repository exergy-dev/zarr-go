import sys, numpy as np
try:
    from zarr.storage import LocalStore
except ImportError:
    from zarr.store import LocalStore
import zarr
arr = zarr.open(LocalStore(sys.argv[1], mode='r'), mode='r')
data = arr[:]
expected = np.load(sys.argv[2])
assert data.shape == expected.shape, f"shape: {data.shape} != {expected.shape}"
assert np.array_equal(data, expected), "values differ"
print(f"OK shape={data.shape} dtype={data.dtype}")
