"""Estimate the dominant eigenvalue with CuPy power iteration."""

from dataclasses import dataclass, replace

import cupy as cp
import numpy as np

from profiling import capture


@dataclass
class PowerIterationConfig:
    dim: int = 1024
    dominance: float = 0.2
    max_steps: int = 300
    check_frequency: int = 25
    residual_threshold: float = 1e-10


def generate_device(cfg):
    cp.random.seed(42)
    weak_lam = cp.random.random(cfg.dim - 1) * (1.0 - cfg.dominance)
    lam = cp.random.permutation(cp.concatenate((cp.array([1.0]), weak_lam)))
    P = cp.random.random((cfg.dim, cfg.dim))
    D = cp.diag(cp.random.permutation(lam))
    return (P @ D) @ cp.linalg.inv(P)


def estimate_device(A, cfg):
    A_gpu = cp.asarray(A)
    x = cp.ones(A_gpu.shape[0], dtype=cp.float64)
    for i in range(0, cfg.max_steps, cfg.check_frequency):
        y = A_gpu @ x
        lam = (x @ y) / (x @ x)
        res = cp.linalg.norm(y - lam * x)
        x = y / cp.linalg.norm(y)
        # Copy checkpoint data to the host.
        cp.asnumpy(x)
        if res < cfg.residual_threshold:
            break
        for _ in range(i + 1, min(i + cfg.check_frequency, cfg.max_steps)):
            y = A_gpu @ x
            x = y / cp.linalg.norm(y)
    return cp.asnumpy((x.T @ (A_gpu @ x)) / (x.T @ x))


def main():
    cfg = PowerIterationConfig()
    A = generate_device(cfg)
    estimate_device(A, replace(cfg, max_steps=2))
    with capture("power_iteration"):
        eigenvalue = estimate_device(A, cfg)
    np.testing.assert_allclose(eigenvalue, 1, atol=1e-4)
    print(f"PASS power_iteration: dominant eigenvalue = {float(eigenvalue):.10f}")


if __name__ == "__main__":
    main()
