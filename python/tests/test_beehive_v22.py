# Copyright © 2026 8b-is

import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mlx.quant as mq
from mlx.quant import (
    HveRlController,
    LatentCompressedAttention,
    ManifoldHyperConnection,
    MultiMaxOutputHead,
    MushroomBodyRouter,
    Ratio,
    ramanujan_bipartite_mask,
    sinkhorn_knopp_mx,
    spectral_gap_report,
)


class TestBeeHiveV22(unittest.TestCase):
    def test_version_and_sinkhorn_mhc_doubly_stochastic(self):
        self.assertEqual(mq.__version__, "3.0.0-alpha.1")

        logits = mx.random.normal((4, 4))
        mat = sinkhorn_knopp_mx(logits, iters=12)
        mx.eval(mat)
        mat_np = np.array(mat)
        np.testing.assert_allclose(mat_np.sum(axis=-1), np.ones(4), atol=1e-4)
        np.testing.assert_allclose(mat_np.sum(axis=-2), np.ones(4), atol=1e-4)
        self.assertTrue(np.all(mat_np >= 0.0))

        class MhcBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mhc = ManifoldHyperConnection(d_model=32, num_streams=4, sinkhorn_iters=6)
                self.proj = nn.Linear(32, 32, bias=False)

            def __call__(self, st):
                return self.mhc(st, self.proj)

        blk = MhcBlock()
        mx.eval(blk.parameters())
        streams = mx.random.normal((2, 8, 4, 32))

        def loss_fn(mod, st):
            return mod(st).square().mean()

        loss, grads = nn.value_and_grad(blk, loss_fn)(blk, streams)
        mx.eval(loss, grads)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertEqual(grads["mhc"]["mix_logits"].shape, (4, 4))
        self.assertTrue((mx.abs(grads["mhc"]["mix_logits"]) > 0).any().item())

    def test_ramanujan_bipartite_mask_and_spectral_gap(self):
        mask = ramanujan_bipartite_mask(out_dim=64, in_dim=128, degree=8, seed=42)
        self.assertEqual(mask.shape, (64, 128))
        row_sums = mask.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.full(64, 8.0))
        col_sums = mask.sum(axis=0)
        self.assertLessEqual(int(col_sums.max() - col_sums.min()), 1)

        rep = spectral_gap_report(mask)
        self.assertTrue(rep["is_expander"])
        self.assertGreater(rep["lambda_1"], rep["lambda_2"])
        self.assertGreater(rep["spectral_gap"], 0.0)
        self.assertAlmostEqual(rep["row_degree"], 8.0)

    def test_latent_compressed_attention_mla_hca_and_gradients(self):
        attn = LatentCompressedAttention(
            d_model=64, n_heads=4, kv_lora_rank=16, hca_stride=8
        )
        mx.eval(attn.parameters())
        self.assertEqual(attn.kv_cache_compression_ratio(), 8.0)

        x = mx.random.normal((2, 24, 64))

        def loss_fn(m, inp):
            return m(inp).square().mean()

        loss, grads = nn.value_and_grad(attn, loss_fn)(attn, x)
        mx.eval(loss, grads)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertTrue((mx.abs(grads["kv_down_proj"]["weight"]) > 0).any().item())
        self.assertTrue((mx.abs(grads["vha_gate"]["weight"]) > 0).any().item())

    def test_mushroom_body_router_wta_habituation_and_quorum(self):
        router = MushroomBodyRouter(
            d_model=32,
            n_kenyon=64,
            kc_degree=6,
            kc_top_k=8,
            num_workers=4,
            top_k_workers=2,
            quorum_threshold=0.05,
        )
        mx.eval(router.parameters())
        h = mx.random.normal((2, 10, 32))

        weights_1, metrics_1 = router(h, update_state=True)
        mx.eval(weights_1, metrics_1["kc_active_density"], metrics_1["mean_habituation"])
        self.assertEqual(weights_1.shape, (2, 10, 4))
        np.testing.assert_allclose(np.array(weights_1.sum(axis=-1)), np.ones((2, 10)), atol=1e-4)
        self.assertLessEqual(float(metrics_1["kc_active_density"].item()), 0.25)

        hab_1 = float(metrics_1["mean_habituation"].item())
        weights_2, metrics_2 = router(h, update_state=True)
        mx.eval(weights_2, metrics_2["mean_habituation"])
        hab_2 = float(metrics_2["mean_habituation"].item())
        self.assertGreater(hab_2, hab_1)

    def test_multimax_head_and_hve_rl_dual_loops(self):
        head = MultiMaxOutputHead(d_model=32, vocab_size=256, num_branches=2)
        mx.eval(head.parameters())
        x = mx.random.normal((2, 6, 32))

        def loss_fn(m, inp):
            return m(inp).square().mean()

        loss, grads = nn.value_and_grad(head, loss_fn)(head, x)
        mx.eval(loss, grads)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertEqual(grads["branch_gate"]["weight"].shape, (2, 32))

        ctrl = HveRlController()
        has_sig, advs = ctrl.compute_tactical_rl_advantages([1.0, 0.0, 0.5])
        self.assertTrue(has_sig)
        self.assertAlmostEqual(sum(advs), 0.0, places=6)

        no_sig, zero_advs = ctrl.compute_tactical_rl_advantages([1.0, 1.0, 1.0])
        self.assertFalse(no_sig)
        self.assertEqual(zero_advs, [0.0, 0.0, 0.0])

        ep = ctrl.evaluate_episode(
            rollout_rewards=[1.0, 0.0],
            logs="cargo test --workspace --locked --offline: 149 passed; 0 failed.\n" * 4,
            tool_calls=["smart_tree(crates/omnilll-core)", "cargo_test(omnilll-core)"],
            reasoning="Verified Ratio GCD and BITCOS parity without allocations.",
            verified_pass=True,
            canary_recall_ratio=Ratio(9, 10),
        )
        self.assertTrue(ep.group_has_signal)
        self.assertGreater(ep.trace_compression_ratio, Ratio(1, 1))
        self.assertGreater(ep.hve_rl_reward, Ratio(1, 1))

        failed_ep = ctrl.evaluate_episode(
            rollout_rewards=[0.0, 0.0],
            logs="compile error",
            tool_calls=[],
            reasoning="bad",
            verified_pass=False,
            canary_recall_ratio=Ratio(9, 10),
        )
        self.assertEqual(failed_ep.hve_rl_reward, Ratio(0, 1))


if __name__ == "__main__":
    unittest.main()
