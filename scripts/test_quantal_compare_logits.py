"""Offline CLI regression tests; no MLX runtime or models required."""

import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("quantal_compare_logits.py")


class ComparisonTests(unittest.TestCase):
    def check_case(self, rust_prompts, ref_prompts, expected):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rust.json").write_text(
                json.dumps({"model_dir": directory, "prompts": rust_prompts})
            )
            (root / "ref.json").write_text(json.dumps({"prompts": ref_prompts}))
            for flags in ([], ["-O"]):
                result = subprocess.run(
                    [
                        sys.executable,
                        *flags,
                        str(SCRIPT),
                        str(root / "rust.json"),
                        str(root / "ref.json"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(
                    result.returncode, expected, result.stdout + result.stderr
                )

    def test_matching(self):
        self.check_case(
            [{"prompt": "hello", "n_tokens": 1, "logits": [1, 2]}],
            [{"prompt": "hello", "n_tokens": 1, "logits_vanilla": [1, 2]}],
            0,
        )

    def test_empty(self):
        self.check_case([], [], 1)

    def test_broadcast(self):
        self.check_case(
            [{"prompt": "hello", "n_tokens": 1, "logits": [1]}],
            [{"prompt": "hello", "n_tokens": 1, "logits_vanilla": [1, 1]}],
            1,
        )

    def test_prompt_mismatch(self):
        self.check_case(
            [{"prompt": "wrong", "n_tokens": 1, "logits": [1]}],
            [{"prompt": "hello", "n_tokens": 1, "logits_vanilla": [1]}],
            1,
        )

    def test_token_mismatch(self):
        self.check_case(
            [{"prompt": "hello", "n_tokens": 2, "logits": [1]}],
            [{"prompt": "hello", "n_tokens": 1, "logits_vanilla": [1]}],
            1,
        )

    def test_nonfinite(self):
        self.check_case(
            [{"prompt": "hello", "n_tokens": 1, "logits": [float("nan")]}],
            [{"prompt": "hello", "n_tokens": 1, "logits_vanilla": [1]}],
            1,
        )

    def test_joint_permutation_preserves_acceptance(self):
        rust = [
            {"prompt": name, "n_tokens": 1, "logits": [value, 0]}
            for name, value in (("alpha", 1), ("beta", 2), ("gamma", 3))
        ]
        ref = [
            {"prompt": row["prompt"], "n_tokens": 1, "logits_vanilla": row["logits"]}
            for row in rust
        ]
        for order in itertools.permutations(range(3)):
            with self.subTest(order=order):
                self.check_case([rust[i] for i in order], [ref[i] for i in order], 0)

    def test_joint_permutation_cannot_hide_failing_item(self):
        rust = [
            {"prompt": name, "n_tokens": 1, "logits": [value, 0]}
            for name, value in (("alpha", 1), ("beta", 9), ("gamma", 3))
        ]
        ref = [
            {"prompt": name, "n_tokens": 1, "logits_vanilla": [value, 0]}
            for name, value in (("alpha", 1), ("beta", 2), ("gamma", 3))
        ]
        for order in itertools.permutations(range(3)):
            with self.subTest(order=order):
                self.check_case([rust[i] for i in order], [ref[i] for i in order], 1)

    def test_one_sided_reorder_rejects_identical_logits(self):
        rust = [
            {"prompt": name, "n_tokens": 1, "logits": [1, 0]}
            for name in ("alpha", "beta")
        ]
        ref = [
            {"prompt": name, "n_tokens": 1, "logits_vanilla": [1, 0]}
            for name in ("beta", "alpha")
        ]
        self.check_case(rust, ref, 1)


if __name__ == "__main__":
    unittest.main()
