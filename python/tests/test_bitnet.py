import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.quant.bitnet import (
    TernaryLinear,
    matmul_ternary_cpu_fallback,
    quantize_ternary_numpy,
    unpack_ternary_numpy,
)


def test_ternary_quantization_and_unpacking_roundtrip():
    np.random.seed(42)
    # Shape [32, 64]
    w = np.random.randn(32, 64).astype(np.float32)
    packed_w, scales = quantize_ternary_numpy(w)

    assert packed_w.shape == (32, 4)  # 64 / 16 = 4 words
    assert scales.shape == (32, 1)

    unpacked_w = unpack_ternary_numpy(packed_w, 64)
    assert unpacked_w.shape == (32, 64)

    # Values in unpacked_w must only be -1, 0, or 1
    unique_vals = set(np.unique(unpacked_w))
    assert unique_vals.issubset({-1, 0, 1})


def test_ternary_linear_layer_forward():
    in_dim = 128
    out_dim = 256
    batch_size = 4

    linear = nn.Linear(in_dim, out_dim)
    ternary_layer = TernaryLinear.from_linear(linear)

    x = mx.random.normal((batch_size, in_dim))
    out = ternary_layer(x)

    assert out.shape == (batch_size, out_dim)
    # Verify non-trivial output
    assert float(mx.sum(mx.abs(out))) > 0.0
