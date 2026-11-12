"""ACH notebook 41: book-byte histograms with global/shared atomics (Apache-2.0)."""

from pathlib import Path
import urllib.request

import cupy as cp
from cuda.coop._experimental import block
from numba import cuda
import numpy as np
import nvtx

from profiling import capture

bins = 256
threads_per_block = 512
items_per_thread = 8
items_per_block = threads_per_block * items_per_thread
block_load = block.make_load(cp.uint8, threads_per_block, items_per_thread, "striped")
BOOKS_URL = "https://drive.usercontent.google.com/download?id=1MW1lPgkTq3YG9ikuq6u3d9sfpt-wKQZ0&export=download"


@cuda.jit
def histogram_global(values, histogram):
    for i in range(items_per_thread):
        value = values[cuda.grid(1) * items_per_thread + i]
        cuda.atomic.add(histogram, value, 1)


@cuda.jit(link=block_load.files)
def histogram_localized(values, histogram):
    items = cuda.local.array(items_per_thread, dtype=values.dtype)
    base = cuda.blockIdx.x * items_per_block
    block_load(values[base:base + items_per_block], items)
    local_histogram = cuda.shared.array(bins, dtype=histogram.dtype)
    for i in range(0, bins, threads_per_block):
        bin = i + cuda.threadIdx.x
        if bin < local_histogram.size:
            local_histogram[bin] = 0
    cuda.syncthreads()
    for i in range(items_per_thread):
        cuda.atomic.add(local_histogram, items[i], 1)
    cuda.syncthreads()
    for i in range(0, bins, threads_per_block):
        bin = i + cuda.threadIdx.x
        if bin < histogram.size:
            cuda.atomic.add(histogram, bin, local_histogram[bin])


def main():
    path = Path("/tmp/books__15m.txt")
    if not path.exists():
        with urllib.request.urlopen(BOOKS_URL, timeout=120) as response:
            path.write_bytes(response.read())
    host = np.fromfile(path, dtype=np.uint8)
    if host.size == 0 or host.size % items_per_block:
        raise ValueError("The tutorial book dataset must be nonempty and a multiple of 4096 bytes")
    values = cp.asarray(host)
    blocks = values.size // items_per_block
    global_result = cp.zeros(bins, dtype=cp.int32)
    local_result = cp.zeros_like(global_result)
    histogram_global[blocks, threads_per_block](values, global_result)
    histogram_localized[blocks, threads_per_block](values, local_result)
    global_result.fill(0)
    local_result.fill(0)
    with capture("histogram"):
        with nvtx.annotate("histogram_global"):
            histogram_global[blocks, threads_per_block](values, global_result)
        with nvtx.annotate("histogram_localized"):
            histogram_localized[blocks, threads_per_block](values, local_result)
    expected = np.bincount(host, minlength=bins)
    np.testing.assert_array_equal(cp.asnumpy(global_result), expected)
    np.testing.assert_array_equal(cp.asnumpy(local_result), expected)
    print(f"PASS histogram: both kernels matched all 256 CPU bins for {host.size:,} bytes")


if __name__ == "__main__":
    main()
