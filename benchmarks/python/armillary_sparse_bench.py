# Copyright © 2026 Peter Lodri / MLX-QUANT Contributors.
# Comprehensive Benchmark: Fused Metal Kernel vs Standard MLX vs Python Baseline

import math
import time

import mlx.core as mx
from mlx.quant.armillary import armillary_sparse_update_metal


def standard_mlx_sparse_update(
    weights,
    grad,
    density,
    phi,
    lr=0.01,
    base_tau=0.005,
    decay_rate=0.001,
    growth_rate=0.05,
):
    """Unfused multi-pass MLX graph execution."""
    sin_phi = mx.sin(phi)
    tau = base_tau * (1.0 + density * mx.abs(sin_phi))
    mask = mx.abs(grad) > tau

    # Weight update
    new_weights = mx.where(mask, weights - lr * grad, weights)

    # Density update
    d_decayed = density * (1.0 - decay_rate)
    new_density = mx.where(mask, d_decayed + growth_rate * mx.abs(grad), d_decayed)

    return new_weights, new_density, mask.astype(mx.float32)


def benchmark_armillary_sparse_update():
    print("=" * 80)
    print(" 🪐 MLX-QUANT ARMILLARY SPHERE FUSED METAL BENCHMARK 🪐")
    print("=" * 80)
    print(
        f"{'Matrix Shape':<18} | {'Unfused MLX (ms)':<18} | {'Fused Metal (ms)':<18} | {'Speedup':<10} | {'Sparsity':<10}"
    )
    print("-" * 80)

    shapes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
    ]

    for rows, cols in shapes:
        shape_str = f"{rows}x{cols}"
        weights = mx.random.normal(shape=(rows, cols)).astype(mx.float32)
        grad = mx.random.normal(shape=(rows, cols)).astype(mx.float32)
        density = mx.abs(weights) * 0.5
        phi = mx.random.uniform(0.0, math.pi, shape=(rows, cols)).astype(mx.float32)

        # Warmup
        for _ in range(5):
            w_u, d_u, m_u = standard_mlx_sparse_update(weights, grad, density, phi)
            mx.eval(w_u, d_u, m_u)
            w_m, d_m, m_m = armillary_sparse_update_metal(weights, grad, density, phi)
            mx.eval(w_m, d_m, m_m)

        # Benchmark Unfused MLX
        iters = 50
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            w_u, d_u, m_u = standard_mlx_sparse_update(weights, grad, density, phi)
            mx.eval(w_u, d_u, m_u)
        mx.synchronize()
        unfused_ms = (time.perf_counter() - t0) * 1000.0 / iters

        # Benchmark Fused Metal
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            w_m, d_m, m_m = armillary_sparse_update_metal(weights, grad, density, phi)
            mx.eval(w_m, d_m, m_m)
        mx.synchronize()
        metal_ms = (time.perf_counter() - t0) * 1000.0 / iters

        speedup = unfused_ms / metal_ms
        sparsity = (1.0 - mx.sum(m_m).item() / m_m.size) * 100.0

        print(
            f"{shape_str:<18} | {unfused_ms:>16.3f}ms | {metal_ms:>16.3f}ms | {speedup:>8.2f}x | {sparsity:>8.2f}%"
        )

    print("=" * 80)


if __name__ == "__main__":
    benchmark_armillary_sparse_update()
