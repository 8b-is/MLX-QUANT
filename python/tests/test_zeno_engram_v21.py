# Copyright © 2026 8b-is

import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mlx.quant as mq
from mlx.nn.layers.bitlinear import (
    BitLinear,
    activation_quant_int4,
    hadamard_wht_mx,
)
from mlx.quant import (
    ASYMMETRIC_CRITICAL_WEIGHT,
    BONOBO_RESONANCE,
    RATIO_ONE,
    EngramMemoryLayer,
    LmHeadLrPolicy,
    Ratio,
    TernaryLinear,
    WsdScheduleConfig,
    bitcos_effective_bpw,
    bitcos_matmul_numpy,
    bitcos_pack_numpy,
    bitcos_unpack_numpy,
    compute_byte_asymmetric_mask,
    evaluate_knowledge_collapse_canaries,
    matmul_ternary_cpu_fallback,
    quantize_ternary_numpy,
)


class TestZenoEngramV21(unittest.TestCase):
    def test_version_and_rational_pq_and_collapse_guard(self):
        self.assertEqual(mq.__version__, "3.0.0-alpha.1")
        self.assertEqual(Ratio(6, 4), BONOBO_RESONANCE)
        self.assertEqual(Ratio(1, 3) + Ratio(1, 6), Ratio(1, 2))
        self.assertEqual(ASYMMETRIC_CRITICAL_WEIGHT, Ratio(3, 1))

        wsd = WsdScheduleConfig()
        self.assertEqual(wsd.lr_at(0).p, 0)
        self.assertEqual(wsd.lr_at(100), wsd.peak_lr)
        self.assertEqual(wsd.lr_at(800), wsd.peak_lr)
        self.assertEqual(wsd.lr_at(1000), wsd.min_lr)
        self.assertTrue(wsd.lr_at(900) < wsd.peak_lr)
        self.assertTrue(wsd.lr_at(900) > wsd.min_lr)

        policy = LmHeadLrPolicy()
        self.assertEqual(
            policy.decoupled_lm_head_lr(850),
            wsd.lr_at(850) * Ratio(1, 5),
        )

        expected = list(range(10))
        healthy = evaluate_knowledge_collapse_canaries(
            expected, [0, 1, 2, 3, 4, 5, 6, 7, 8, 99], 64, 1_000_000, policy
        )
        self.assertFalse(healthy.collapse_detected)
        self.assertEqual(healthy.recall_ratio, Ratio(9, 10))

        collapsed = evaluate_knowledge_collapse_canaries(
            expected, [0, 99, 99, 99, 99, 99, 99, 99, 99, 99], 64, 1_000_000, policy
        )
        self.assertTrue(collapsed.collapse_detected)
        self.assertEqual(collapsed.recall_ratio, Ratio(1, 10))

        mask = compute_byte_asymmetric_mask("#![no_std] fn x() {}")
        self.assertEqual(mask[3], 3.0)
        self.assertEqual(mask[-1], 1.0)

    def test_bitcos_pack_unpack_matmul_and_ternary_gpu_parity(self):
        rng = np.random.default_rng(7)
        # Construct sparse weight matrix (> 55% inside deadzone -> < 1.585 bpw)
        w = rng.normal(0.0, 1.0, size=(8, 67)).astype(np.float32)
        zero_mask = rng.random(w.shape) < 0.58
        w[zero_mask] = 0.01 * w[zero_mask]

        packed = bitcos_pack_numpy(w)
        self.assertLess(packed.effective_bpw, 1.585)
        self.assertEqual(
            bitcos_effective_bpw(1000, 485),
            1.485,
        )

        unpacked = bitcos_unpack_numpy(packed)
        self.assertEqual(unpacked.shape, (8, 67))
        self.assertTrue(set(np.unique(unpacked)).issubset({-1, 0, 1}))

        x = rng.normal(0.0, 1.0, size=(3, 67)).astype(np.float32)
        y_bitcos = bitcos_matmul_numpy(x, packed)
        y_manual = np.dot(x, unpacked.astype(np.float32).T) * packed.scales.T
        np.testing.assert_allclose(y_bitcos, y_manual, rtol=1e-5, atol=1e-5)

        # Verify native Metal GPU TernaryLinear matches CPU fallback
        lin = nn.Linear(67, 16, bias=True)
        mx.eval(lin.parameters())
        tlin = TernaryLinear.from_linear(lin)
        x_mx = mx.array(x)
        y_gpu = tlin(x_mx)
        mx.eval(y_gpu)

        w_np = np.array(lin.weight.astype(mx.float32))
        packed_u32, scales = quantize_ternary_numpy(w_np)
        y_cpu = matmul_ternary_cpu_fallback(x, packed_u32, scales, 67) + np.array(lin.bias)
        np.testing.assert_allclose(np.array(y_gpu), y_cpu, rtol=1e-4, atol=1e-4)

    def test_engram_lookup_gating_and_gradients(self):
        engram = EngramMemoryLayer(
            d_model=32,
            vocab_size=256,
            table_size=127,
            num_heads=4,
            ngram_orders=(2, 3),
            head_dim=8,
        )
        mx.eval(engram.parameters())
        h = mx.random.normal((2, 12, 32))
        toks = mx.array(
            [[35, 33, 91, 110, 111, 95, 115, 116, 100, 93, 32, 10]] * 2,
            dtype=mx.uint32,
        )

        def loss_fn(model, hidden, ids):
            return model(hidden, ids).square().sum()

        loss, grads = nn.value_and_grad(engram, loss_fn)(engram, h, toks)
        mx.eval(loss, grads)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertTrue((mx.abs(grads["table"]["weight"]) > 0).any().item())
        self.assertTrue((mx.abs(grads["value_proj"]["weight"]) > 0).any().item())

    def test_bitlinear_tequila_and_hadamard_int4(self):
        spike = mx.zeros((2, 16), dtype=mx.float32)
        spike[:, 0] = 16.0
        spike[:, 1] = -2.0
        h_spike = hadamard_wht_mx(spike)
        mx.eval(h_spike)
        self.assertLess(
            float(mx.max(mx.abs(h_spike)).item()),
            float(mx.max(mx.abs(spike)).item()),
        )
        q4 = activation_quant_int4(h_spike)
        mx.eval(q4)
        self.assertEqual(q4.shape, spike.shape)

        layer = BitLinear(
            16,
            8,
            bias=True,
            deployed_forward=True,
            tequila_lambda=1e-2,
            use_hadamard=True,
            act_bits=4,
        )
        # Force deadzone weights inside [-0.1, 0.1] while keeping active weights at +-2.0
        w_init = np.zeros((8, 16), dtype=np.float32)
        w_init[:, :4] = 2.0
        w_init[:, 4:8] = -2.0
        w_init[:, 8:] = 0.05
        layer.weight = mx.array(w_init)
        mx.eval(layer.parameters())

        # Pass zero inputs on deadzone columns [8..16]: Tequila MUST still deliver gradients!
        x = mx.zeros((3, 16), dtype=mx.float32)
        x[:, :8] = 1.0

        def loss_fn(m, inp):
            return m(inp).sum()

        _, grads = nn.value_and_grad(layer, loss_fn)(layer, x)
        mx.eval(grads)
        deadzone_grad = np.array(grads["weight"])[:, 8:]
        self.assertTrue(np.all(np.abs(deadzone_grad) > 0.0))


if __name__ == "__main__":
    unittest.main()
