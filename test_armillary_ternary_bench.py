import math
import time
import unittest

import mlx.core as mx

from armillary_sphere_tensor import (
    ArmillarySphereTensor,
    metal_packed_ternary_matvec,
    pack_ternary_2bit,
    quantize_ternary_groupwise,
    quantize_ternary_naive,
    quantize_ternary_optimal_groupwise,
    unpack_ternary_2bit,
)


class TestArmillaryTernaryAndMetalBench(unittest.TestCase):
    PACKED_SPEEDUP_FLOOR = 1.0

    def test_optimal_groupwise_ternary_beats_naive_mse_and_ppl(self):
        mx.random.seed(42)
        # Simulate realistic heavy-tailed / multi-scale linear weights (1024 x 1024)
        base = mx.random.normal(shape=(1024, 1024))
        row_scales = mx.exp(mx.linspace(-1.2, 1.2, 1024)).reshape(1024, 1)
        w = base * row_scales
        x = mx.random.normal(shape=(32, 1024)) / math.sqrt(1024)
        y_ref = x @ w.T
        mx.eval(w, x, y_ref)

        q_naive, _, w_naive = quantize_ternary_naive(w)
        q_opt, _, w_opt = quantize_ternary_optimal_groupwise(
            w, group_size=64, threshold_factor=0.68, refinement_steps=0
        )
        q_ref, _, w_ref = quantize_ternary_groupwise(
            w, group_size=64, threshold_factor=0.68, refinement_steps=2
        )
        mx.eval(w_naive, w_opt, w_ref)

        mse_naive = mx.mean((w - w_naive) ** 2).item()
        mse_opt = mx.mean((w - w_opt) ** 2).item()
        mse_ref = mx.mean((w - w_ref) ** 2).item()

        y_naive = x @ w_naive.T
        y_opt = x @ w_opt.T
        y_refined = x @ w_ref.T
        mx.eval(y_naive, y_opt, y_refined)

        logit_mse_naive = mx.mean((y_ref - y_naive) ** 2).item()
        logit_mse_opt = mx.mean((y_ref - y_opt) ** 2).item()
        logit_mse_ref = mx.mean((y_ref - y_refined) ** 2).item()
        ppl_proxy_naive = math.exp(logit_mse_naive)
        ppl_proxy_opt = math.exp(logit_mse_opt)
        ppl_proxy_ref = math.exp(logit_mse_ref)

        # Also test heavy-tailed intra-group outlier channel distribution
        w_heavy = (base**3) * 0.25
        _, _, wh_naive = quantize_ternary_naive(w_heavy)
        _, _, wh_opt1 = quantize_ternary_optimal_groupwise(
            w_heavy, group_size=64, refinement_steps=0
        )
        _, _, wh_ref2 = quantize_ternary_groupwise(
            w_heavy, group_size=64, refinement_steps=2
        )
        mx.eval(wh_naive, wh_opt1, wh_ref2)
        hmse_naive = mx.mean((w_heavy - wh_naive) ** 2).item()
        hmse_opt1 = mx.mean((w_heavy - wh_opt1) ** 2).item()
        hmse_ref2 = mx.mean((w_heavy - wh_ref2) ** 2).item()

        print(
            f"\n[mlxQuant Receipt] Multi-Scale Weight MSE: naive={mse_naive:.6f} -> "
            f"1-step_groupwise={mse_opt:.6f} -> 2-step_refined={mse_ref:.6f} "
            f"({(1.0 - mse_ref / mse_naive) * 100:.2f}% MSE reduction vs naive)"
        )
        print(
            f"[mlxQuant Receipt] Heavy-Tailed Outlier Weight MSE: naive={hmse_naive:.6f} -> "
            f"1-step_groupwise={hmse_opt1:.6f} -> 2-step_refined={hmse_ref2:.6f} "
            f"({(1.0 - hmse_ref2 / hmse_opt1) * 100:.2f}% MSE reduction vs 1-step)"
        )
        print(
            f"[mlxQuant Receipt] Perplexity Proxy exp(LogitMSE): naive={ppl_proxy_naive:.4f} -> "
            f"1-step={ppl_proxy_opt:.4f} -> 2-step_refined={ppl_proxy_ref:.4f}"
        )
        self.assertLess(mse_opt, mse_naive * 0.65)
        self.assertLess(mse_ref, mse_opt)
        self.assertLess(hmse_ref2, hmse_opt1 * 0.60)
        self.assertLess(ppl_proxy_ref, ppl_proxy_opt)

    def test_2bit_packed_ternary_roundtrip_and_8x_fp16_compression(self):
        mx.random.seed(11)
        w_fp16 = mx.random.normal(shape=(1024, 1024)).astype(mx.float16)
        q, scales, w_recon = quantize_ternary_groupwise(
            w_fp16.astype(mx.float32), group_size=64
        )
        w_packed = pack_ternary_2bit(q)
        q_unpacked = unpack_ternary_2bit(w_packed, dtype=mx.float32)
        w_unpacked = unpack_ternary_2bit(
            w_packed, scales=scales, group_size=64, dtype=mx.float32
        )
        mx.eval(w_packed, q_unpacked, w_unpacked)

        fp16_bytes = w_fp16.nbytes
        int8_bytes = q.astype(mx.int8).nbytes
        packed_bytes = w_packed.nbytes
        ratio_vs_fp16 = fp16_bytes / packed_bytes
        ratio_vs_int8 = int8_bytes / packed_bytes

        print(
            f"\n[mlxQuant Receipt] 2-Bit Packed Ternary Storage: fp16={fp16_bytes / 1024:.1f} KB, "
            f"int8={int8_bytes / 1024:.1f} KB -> uint8_2bit={packed_bytes / 1024:.1f} KB "
            f"({ratio_vs_fp16:.2f}x vs FP16, {ratio_vs_int8:.2f}x vs INT8)"
        )
        self.assertEqual(w_packed.dtype, mx.uint8)
        self.assertAlmostEqual(ratio_vs_fp16, 8.0, places=4)
        self.assertAlmostEqual(ratio_vs_int8, 4.0, places=4)
        self.assertEqual(mx.max(mx.abs(q - q_unpacked)).item(), 0.0)
        self.assertLess(mx.max(mx.abs(w_recon - w_unpacked)).item(), 1e-6)

    def test_metal_packed_ternary_matvec_accuracy_and_bandwidth(self):
        mx.random.seed(19)
        N, K = 4096, 4096
        w = mx.random.normal(shape=(N, K)).astype(mx.float16)
        q, scales, _ = quantize_ternary_groupwise(w.astype(mx.float32), group_size=64)
        w_packed = pack_ternary_2bit(q)
        scales_fp16 = scales.astype(mx.float16).reshape(N, K // 64)
        x_1d = mx.random.normal(shape=(K,)).astype(mx.float16)
        x_2d = mx.random.normal(shape=(4, K)).astype(mx.float16)
        w_deq_fp16 = unpack_ternary_2bit(
            w_packed, scales=scales_fp16, group_size=64, dtype=mx.float16
        )
        mx.eval(w_packed, scales_fp16, x_1d, x_2d, w_deq_fp16)

        y_ref_1d = x_1d @ w_deq_fp16.T
        y_metal_1d = metal_packed_ternary_matvec(
            x_1d, w_packed, scales_fp16, group_size=64
        )
        y_ref_2d = x_2d @ w_deq_fp16.T
        y_metal_2d = metal_packed_ternary_matvec(
            x_2d, w_packed, scales_fp16, group_size=64
        )
        mx.eval(y_ref_1d, y_metal_1d, y_ref_2d, y_metal_2d)

        max_err_1d = mx.max(
            mx.abs(y_ref_1d.astype(mx.float32) - y_metal_1d.astype(mx.float32))
        ).item()
        max_err_2d = mx.max(
            mx.abs(y_ref_2d.astype(mx.float32) - y_metal_2d.astype(mx.float32))
        ).item()
        self.assertLess(max_err_1d, 0.15)
        self.assertLess(max_err_2d, 0.15)

        # Warmup
        for _ in range(5):
            mx.eval(
                unpack_ternary_2bit(
                    w_packed, scales=scales_fp16, group_size=64, dtype=mx.float16
                )
                @ x_1d
            )
            mx.eval(x_1d @ w_deq_fp16.T)
            mx.eval(
                metal_packed_ternary_matvec(x_1d, w_packed, scales_fp16, group_size=64)
            )

        iters = 40
        unpack_matmul_ms = self._best_ms(
            lambda: mx.eval(
                x_1d
                @ unpack_ternary_2bit(
                    w_packed, scales=scales_fp16, group_size=64, dtype=mx.float16
                ).T
            ),
            iters,
        )
        dense_fp16_ms = self._best_ms(lambda: mx.eval(x_1d @ w_deq_fp16.T), iters)
        metal_packed_ms = self._best_ms(
            lambda: mx.eval(
                metal_packed_ternary_matvec(x_1d, w_packed, scales_fp16, group_size=64)
            ),
            iters,
        )

        packed_bytes_transferred = w_packed.nbytes + scales_fp16.nbytes + x_1d.nbytes
        fp16_equiv_bytes = w_deq_fp16.nbytes + x_1d.nbytes
        packed_gbps = (packed_bytes_transferred / 1e9) / (metal_packed_ms / 1000.0)
        eff_fp16_gbps = (fp16_equiv_bytes / 1e9) / (metal_packed_ms / 1000.0)

        print(
            f"\n[mac++ Receipt] 4096x4096 MatVec Latency: unpack+matmul={unpack_matmul_ms:.3f} ms | "
            f"dense_fp16={dense_fp16_ms:.3f} ms | fused_metal_2bit={metal_packed_ms:.3f} ms "
            f"({unpack_matmul_ms / metal_packed_ms:.2f}x vs unpack+matmul, "
            f"{dense_fp16_ms / metal_packed_ms:.2f}x vs dense FP16)"
        )
        print(
            f"[mac++ Receipt] Fused Metal 2-Bit Bandwidth: physical={packed_gbps:.2f} GB/s "
            f"| effective_fp16_equiv={eff_fp16_gbps:.2f} GB/s | max_err_1d={max_err_1d:.5f}"
        )
        self.assertLess(metal_packed_ms, unpack_matmul_ms * self.PACKED_SPEEDUP_FLOOR)

    @staticmethod
    def _best_ms(fn, iters, repeats=5):
        best = math.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            best = min(best, (time.perf_counter() - t0) * 1000.0 / iters)
        return best

    def test_apple_silicon_metal_compiled_and_ternary_speedup(self):
        mx.random.seed(7)
        sphere = ArmillarySphereTensor(
            in_features=2048, out_features=2048, group_size=64
        )
        grad = mx.random.normal(shape=(2048, 2048))
        x = mx.random.normal(shape=(16, 2048))
        mx.eval(sphere.weights, sphere.density_field, grad, x)

        # Warmup both paths
        _ = sphere.sparse_recursive_update(grad, lr=0.01, base_tau=1.2)
        _ = sphere.sparse_recursive_update_compiled(grad, lr=0.01, base_tau=1.2)
        _ = mx.eval(sphere.forward(x))
        _ = mx.eval(sphere.forward_ternary_metal(x))

        iters = 40
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = sphere.sparse_recursive_update(grad, lr=0.001, base_tau=1.2)
        eager_ms = (time.perf_counter() - t0) * 1000.0 / iters

        t1 = time.perf_counter()
        for _ in range(iters):
            _ = sphere.sparse_recursive_update_compiled(grad, lr=0.001, base_tau=1.2)
        compiled_ms = (time.perf_counter() - t1) * 1000.0 / iters

        fp32_bytes = sphere.weights.nbytes
        ternary_bytes = sphere.w_q.nbytes + sphere.scales.nbytes
        compression_ratio = fp32_bytes / ternary_bytes

        print(
            f"\n[mac++ Receipt] Device={mx.default_device()} | Geodesic Update: eager={eager_ms:.3f} ms/iter -> "
            f"@mx.compile={compiled_ms:.3f} ms/iter ({eager_ms / compiled_ms:.2f}x speedup)"
        )
        print(
            f"[mac++ Receipt] Metal Ternary Memory: fp32={fp32_bytes / (1024*1024):.2f} MB -> "
            f"packed_ternary={ternary_bytes / (1024*1024):.4f} MB ({compression_ratio:.2f}x compression)"
        )
        self.assertAlmostEqual(compression_ratio, 12.80, places=2)
        self.assertLess(compiled_ms, eager_ms)


if __name__ == "__main__":
    unittest.main(verbosity=2)
