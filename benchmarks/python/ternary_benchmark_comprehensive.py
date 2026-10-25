# Copyright © 2026 8b-is
#
# BitNet b1.58 Ternary Metal Kernel Benchmark & Memory Bandwidth Profiler
# Measures latency, throughput (tokens/sec), effective bandwidth, and memory savings
# across M-series unified memory architecture.

import time

import mlx.core as mx
import mlx.nn as nn


def run_benchmark():
    if not mx.metal.is_available():
        raise SystemExit(
            "Metal GPU not available. This benchmark requires Apple Silicon."
        )

    print("=" * 80)
    print("✦ MLX-QUANT: Zero-Copy BitNet b1.58 Ternary Metal Kernel Benchmark")
    print(f"✦ Device: Apple Silicon (Unified Memory Architecture)")
    print("=" * 80)

    shapes = [
        # (Name, M, K, N)
        ("Llama-3-8B QKV Projection (Decode M=1)", 1, 4096, 4096),
        ("Llama-3-8B FFN Up-Projection (Decode M=1)", 1, 4096, 14336),
        ("Llama-3-8B FFN Down-Projection (Decode M=1)", 1, 14336, 4096),
        ("Llama-3-8B Decode Prefill (M=32)", 32, 4096, 4096),
        ("Llama-3-8B Decode Prefill (M=64)", 64, 4096, 4096),
        ("Llama-3-8B Batched Decode (M=8, K=8192, N=4096)", 8, 8192, 4096),
        ("70B Scale Projection (Decode M=1)", 1, 8192, 28672),
    ]

    group_size = 64
    bits = 2

    header = f"{'Workload / Shape':<45} | {'Dense FP32':<11} | {'Compose':<11} | {'Fused 1.58b':<11} | {'Speedup':<8} | {'Bandwidth'}"
    print(header)
    print("-" * len(header))

    WARMUP = 15
    ITERS = 100

    for name, M, K, N in shapes:
        mx.random.seed(0)
        x = mx.random.normal((M, K)).astype(mx.float32)
        w = mx.random.normal((N, K)).astype(mx.float32)

        # Quantize to ternary
        w_q, scales = mx.quantize(w, group_size=group_size, bits=bits, mode="ternary")
        mx.eval(x, w, w_q, scales)

        # 1. Fused Ternary Metal Kernel
        def run_fused():
            return mx.quantized_matmul(
                x,
                w_q,
                scales,
                group_size=group_size,
                bits=bits,
                mode="ternary",
                transpose=True,
            )

        # 2. Compose (Dequantize + Dense Matmul)
        def run_compose():
            w_dense = mx.dequantize(
                w_q, scales, group_size=group_size, bits=bits, mode="ternary"
            )
            return x @ w_dense.T

        # 3. Dense FP32
        def run_dense():
            return x @ w.T

        # Warmup
        for _ in range(WARMUP):
            res_fused = run_fused()
            res_compose = run_compose()
            res_dense = run_dense()
            mx.eval(res_fused, res_compose, res_dense)

        # Measure Fused
        t0 = time.perf_counter()
        for _ in range(ITERS):
            res = run_fused()
            mx.eval(res)
        t_fused = (time.perf_counter() - t0) / ITERS * 1000.0  # ms

        # Measure Compose
        t0 = time.perf_counter()
        for _ in range(ITERS):
            res = run_compose()
            mx.eval(res)
        t_compose = (time.perf_counter() - t0) / ITERS * 1000.0  # ms

        # Measure Dense
        t0 = time.perf_counter()
        for _ in range(ITERS):
            res = run_dense()
            mx.eval(res)
        t_dense = (time.perf_counter() - t0) / ITERS * 1000.0  # ms

        # Memory transferred for weight read in ternary vs FP32
        ternary_bytes = (N * K * bits / 8) + (N * (K / group_size) * 4)
        effective_gb_s = (ternary_bytes / (t_fused / 1000.0)) / (1024**3)
        speedup = f"{t_compose / t_fused:.2f}x"

        row = f"{name:<45} | {t_dense:>8.3f} ms | {t_compose:>8.3f} ms | {t_fused:>8.3f} ms | {speedup:>8} | {effective_gb_s:>7.1f} GB/s"
        print(row)

    print("=" * 80)
    print("✦ Memory Footprint Summary (Llama-3-8B Linear 4096x4096):")
    dense_mb = (4096 * 4096 * 4) / (1024 * 1024)
    ternary_mb = ((4096 * 4096 * 2 / 8) + (4096 * 64 * 4)) / (1024 * 1024)
    print(f"  • FP32 Unquantized: {dense_mb:.2f} MB")
    print(f"  • Packed Ternary 1.58-bit (weights + scales): {ternary_mb:.2f} MB")
    print(
        f"  • Compression: {dense_mb / ternary_mb:.2f}x ({100 * (1 - ternary_mb/dense_mb):.1f}% VRAM saved)"
    )
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
