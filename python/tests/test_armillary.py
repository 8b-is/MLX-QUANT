import math

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.quant.armillary import ArmillarySphereLinear, armillary_sparse_update_metal


def test_armillary_metal_kernel_direct():
    """Verify raw Metal Shading Language kernel for Armillary Sparse Recursion."""
    shape = (64, 128)
    weights = mx.ones(shape, dtype=mx.float32)
    # Gradients: half high magnitude, half zero
    grad = mx.concatenate(
        [mx.ones((32, 128)) * 2.0, mx.zeros((32, 128))], axis=0
    ).astype(mx.float32)
    density = mx.zeros(shape, dtype=mx.float32)
    phi = mx.ones(shape, dtype=mx.float32) * (math.pi / 2.0)

    out_w, out_d, mask = armillary_sparse_update_metal(
        weights=weights,
        grad=grad,
        density=density,
        phi=phi,
        lr=0.1,
        base_tau=0.5,
        decay_rate=0.01,
        growth_rate=0.05,
    )

    mx.eval(out_w, out_d, mask)

    assert out_w.shape == shape
    assert out_d.shape == shape
    assert mask.shape == shape

    # Top half should be updated: 1.0 - 0.1 * 2.0 = 0.8
    assert mx.allclose(out_w[:32, :], mx.ones((32, 128)) * 0.8)
    # Bottom half should be untouched: 1.0
    assert mx.allclose(out_w[32:, :], mx.ones((32, 128)) * 1.0)
    # Mask check
    assert mx.allclose(mask[:32, :], mx.ones((32, 128)))
    assert mx.allclose(mask[32:, :], mx.zeros((32, 128)))


def test_armillary_sphere_linear_layer():
    """Verify ArmillarySphereLinear forward pass and sparse recursive update."""
    in_features = 256
    out_features = 512
    batch_size = 8

    layer = ArmillarySphereLinear(
        in_features=in_features, out_features=out_features, base_tau=0.1
    )

    x = mx.random.normal(shape=(batch_size, in_features))
    y = layer(x)
    mx.eval(y)

    assert y.shape == (batch_size, out_features)

    # Simulate gradient
    grad = mx.random.normal(shape=(out_features, in_features))
    sparsity, energy = layer.update_sparse_metal(grad=grad, lr=0.05)

    assert 0.0 <= sparsity <= 1.0
    assert energy >= 0.0

    # Steganographic extraction test
    stego_field = layer.extract_steganographic_field()
    mx.eval(stego_field)
    assert stego_field.size > 0
