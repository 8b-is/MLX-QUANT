import math
import time
import unittest
import mlx.core as mx
from armillary_sphere_tensor import (
    ArmillarySphereTensor,
    quantize_ternary_naive,
    quantize_ternary_optimal_groupwise,
)


class TestArmillaryTernaryAndMetalBench(unittest.TestCase):
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
        q_opt, _, w_opt = quantize_ternary_optimal_groupwise(w, group_size=64, threshold_factor=0.68)
        mx.eval(w_naive, w_opt)

        mse_naive = mx.mean((w - w_naive) ** 2).item()
        mse_opt = mx.mean((w - w_opt) ** 2).item()

        y_naive = x @ w_naive.T
        y_opt = x @ w_opt.T
        mx.eval(y_naive, y_opt)

        logit_mse_naive = mx.mean((y_ref - y_naive) ** 2).item()
        logit_mse_opt = mx.mean((y_ref - y_opt) ** 2).item()
        ppl_proxy_naive = math.exp(logit_mse_naive)
        ppl_proxy_opt = math.exp(logit_mse_opt)

        print(
            f"\n[mlxQuant Receipt] Weight MSE: naive={mse_naive:.6f} -> optimal_groupwise={mse_opt:.6f} "
            f"({(1.0 - mse_opt / mse_naive) * 100:.2f}% MSE reduction)"
        )
        print(
            f"[mlxQuant Receipt] Perplexity Proxy exp(LogitMSE): naive={ppl_proxy_naive:.4f} -> optimal={ppl_proxy_opt:.4f}"
        )
        self.assertLess(mse_opt, mse_naive * 0.65)
        self.assertLess(ppl_proxy_opt, ppl_proxy_naive)

    def test_apple_silicon_metal_compiled_and_ternary_speedup(self):
        mx.random.seed(7)
        sphere = ArmillarySphereTensor(in_features=2048, out_features=2048, group_size=64)
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
