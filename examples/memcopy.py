"""ACH notebook 40: blocked and coalesced device copy kernels (Apache-2.0)."""

import cupy as cp
from numba import cuda
import nvtx

from profiling import capture


@cuda.jit
def copy_blocked(src, dst, items_per_thread):
    base = cuda.grid(1) * items_per_thread
    for i in range(items_per_thread):
        dst[base + i] = src[base + i]


@cuda.jit
def copy_optimized(src, dst, items_per_thread):
    bd = cuda.blockDim.x
    bx = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    items_per_block = bd * items_per_thread
    base = tx + bx * items_per_block
    for i in range(0, items_per_block, bd):
        dst[base + i] = src[base + i]


def main():
    threads_per_block, items_per_thread, total_items = 256, 64, 2**24
    blocks = total_items // (threads_per_block * items_per_thread)
    src = cp.arange(total_items, dtype=cp.int64)
    blocked, optimized = cp.empty_like(src), cp.empty_like(src)
    copy_blocked[blocks, threads_per_block](src, blocked, items_per_thread)
    copy_optimized[blocks, threads_per_block](src, optimized, items_per_thread)
    blocked.fill(-1)
    optimized.fill(-1)
    with capture("memcopy"):
        with nvtx.annotate("copy_blocked"):
            copy_blocked[blocks, threads_per_block](src, blocked, items_per_thread)
        with nvtx.annotate("copy_optimized"):
            copy_optimized[blocks, threads_per_block](src, optimized, items_per_thread)
    cp.testing.assert_array_equal(src, blocked)
    cp.testing.assert_array_equal(src, optimized)
    print(f"PASS memcopy: both kernels copied {src.nbytes:,} bytes correctly")


if __name__ == "__main__":
    main()
