"""Bracket warmed-up work with CUDA profiler controls and an NVTX label."""

from contextlib import contextmanager

import cupy as cp
import nvtx


@contextmanager
def capture(name):
    cp.cuda.Device().synchronize()
    cp.cuda.profiler.start()
    try:
        with nvtx.annotate(name):
            yield
            cp.cuda.Device().synchronize()
    finally:
        cp.cuda.profiler.stop()
