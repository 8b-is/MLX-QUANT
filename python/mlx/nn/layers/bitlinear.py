# Copyright © 2026 8b-is
#
# BitNet b1.58 (arXiv:2402.17764) ternary/binary weight simulation for MLX.
#
# This is a quantization-*aware training* layer, not a packed quantized
# format: weights stay full-precision `array`s and are re-quantized on every
# forward pass via a straight-through estimator (the forward pass sees the
# quantized value; `mx.stop_gradient` makes the backward pass treat that
# quantization as identity, so gradients still reach the full-precision
# weights the optimizer actually updates). It does not reduce memory the way
# `nn.QuantizedLinear` does -- there is no ternary-packed storage or a fused
# Metal kernel here. That is real, harder, follow-on work; this lays the
# groundwork with a version that is honestly scoped and already useful for
# training BitNet-style models in MLX today.

from functools import partial
from typing import Optional

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.nn.layers.normalization import RMSNorm


@partial(mx.compile, shapeless=True)
def activation_quant(x: mx.array) -> mx.array:
    r"""Per-token simulated int8 activation quantization.

    .. math::

        s = \frac{127}{\max(|x|, \text{dim}=-1)} \qquad
        y = \frac{\text{round}(\text{clip}(x \cdot s, -128, 127))}{s}
    """
    scale = 127.0 / mx.clip(mx.abs(x).max(axis=-1, keepdims=True), 1e-5, None)
    return mx.clip(mx.round(x * scale), -128, 127) / scale


@partial(mx.compile, shapeless=True)
def activation_quant_int4(x: mx.array) -> mx.array:
    """Per-token simulated symmetric INT4 activation quantization (`[-7, +7]`) for `H-BitLinear`."""
    abs_max = mx.clip(mx.abs(x).max(axis=-1, keepdims=True), 1e-5, None)
    scale = 7.0 / abs_max
    return mx.clip(mx.round(x * scale), -7.0, 7.0) / scale


def hadamard_wht_mx(x: mx.array) -> mx.array:
    """
    Orthonormal Fast Walsh-Hadamard Transform (`WHT(x) / sqrt(d)`) on power-of-2 last dim
    for BitNet v2 `H-BitLinear` (`arXiv:2504.18415`).
    """
    d = x.shape[-1]
    if d == 0 or (d & (d - 1)) != 0:
        raise ValueError(f"hadamard_wht_mx requires power-of-2 last dim, got {d}")
    out = x
    h = 1
    while h < d:
        reshaped = out.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        a = reshaped[..., 0, :]
        b = reshaped[..., 1, :]
        out = mx.concatenate([a + b, a - b], axis=-1).reshape(x.shape)
        h *= 2
    return out * (d ** -0.5)


def weight_quant(w: mx.array, threshold: float = 0.5, group_size: int = 64) -> mx.array:
    r"""True thresholded ternary weight quantization (BitNet b1.58).

    .. math::

        \alpha_g = \mathrm{mean}_{j \in g}(|w|), \qquad
        \hat{w}_j = \mathrm{round}(w_j / \alpha_g) \cdot \alpha_g

    per ``group_size``-column group (matches the ayeOS scale layout the Rust
    runner dequantizes as ``(code-1) * scale``). Values with
    :math:`|w_j| < \frac{1}{2}\alpha_g` land on the true third state,
    **zero**. Measured on the trained SOTA checkpoint ~32-34% of the weights
    sit below ``0.5 * scale`` and want the zero state; ``sign``-based
    collapse (the old behaviour) pushed them to ``±scale`` and wasted the
    format's third code. This is the quantizer the deployed forward and the
    ayeOS export share, so training ≡ deployment ≡ Rust by construction.

    ``threshold`` scales the zero band as a fraction of ``scale``; the default
    ``0.5`` is the `round` tie point (BitNet's own rule).
    """
    if group_size is None or w.ndim < 2 or w.shape[-1] < group_size or (w.shape[-1] % group_size) != 0:
        scale = mx.abs(w).mean(axis=-1, keepdims=True) if w.ndim >= 2 else mx.abs(w).mean()
        return mx.where(mx.abs(w) < threshold * scale, 0.0, mx.sign(w) * scale)
    orig_shape = w.shape
    k = orig_shape[-1]
    n = 1
    for d in orig_shape[:-1]:
        n *= d
    w2 = mx.reshape(w, (n, k))
    wr = mx.reshape(w2, (n, k // group_size, group_size))
    scale_g = mx.abs(wr).mean(axis=-1, keepdims=True)
    q = mx.where(mx.abs(wr) < threshold * scale_g, 0.0, mx.sign(wr) * scale_g)
    return mx.reshape(q, orig_shape)


@partial(mx.compile, shapeless=True)
def binary_weight_quant(w: mx.array) -> mx.array:
    """Provably 1-bit: every element is exactly ``+scale`` or ``-scale``, no
    exceptions. Unlike :func:`weight_quant`'s ``sign`` (which *could* in
    principle return exactly 0 for ``w == mean(w)``), the ``where`` boundary
    case is deterministic -- ties round to +1, not to an ambiguous third
    value.
    """
    scale = mx.abs(w).mean()
    shifted = w - w.mean()
    return mx.where(shifted >= 0, scale, -scale)


@partial(mx.compile, shapeless=True)
def _quantize_ste(full: mx.array, quantized: mx.array) -> mx.array:
    """Straight-through estimator, as its own compiled op rather than glue
    code in ``__call__``: forward returns ``quantized``; backward treats it
    as identity, so gradients reach ``full`` unchanged. Compiling this
    subtract/stop_gradient/add sequence separately fuses it into one
    dispatched kernel on the hot path instead of three uncompiled ops."""
    return full + mx.stop_gradient(quantized - full)


class BitLinear(Module):
    r"""A :obj:`~mlx.nn.Linear` with BitNet b1.58 weight quantization applied
    on every forward pass via a straight-through estimator.

    Concretely, forward uses the quantized weight and activation; backward
    treats both quantization steps as identity, so gradients reach the
    full-precision ``weight`` the optimizer actually updates:

    .. math::

        y = \hat{x} \hat{W}^\top + b

    where :math:`\hat{x}` = :func:`activation_quant` (x) and :math:`\hat{W}`
    = :func:`weight_quant` (W), or :func:`binary_weight_quant` if
    ``binary=True``.

    Args:
        input_dims (int): The dimensionality of the input features.
        output_dims (int): The dimensionality of the output features.
        bias (bool, optional): If ``False`` the layer has no bias.
            Default: ``True``.
        binary (bool, optional): Use :func:`binary_weight_quant` (provably
            1-bit) instead of :func:`weight_quant` (nominally ternary,
            measured ~binary). Default: ``False``.
        norm_eps (float, optional): ``eps`` for the input :obj:`RMSNorm`.
            Default: ``1e-6``.
        deployed_forward (bool, optional): Weight-quant-only forward —
            ``x @ weight_quant(W).T``, NO per-projection RMSNorm, NO
            activation_quant. This is the deployed form (what the Rust
            runner implements). ``False`` enables the faithful BitNet
            b1.58 forward (per-projection RMSNorm + int8 activation_quant)
            used by the pre-``deployed-forward`` checkpoints. The ``norm``
            parameter is only created when ``deployed_forward=False``, so
            deployed checkpoints carry no per-projection gain archaeology.
            Default: ``True``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        binary: bool = False,
        norm_eps: float = 1e-6,
        deployed_forward: bool = True,
        tequila_lambda: float = 0.0,
        use_hadamard: bool = False,
        act_bits: int = 8,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        scale = (1.0 / input_dims) ** 0.5
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims),
        )
        if bias:
            self.bias = mx.zeros((output_dims,))
        self.binary = binary
        self.deployed_forward = deployed_forward
        self.tequila_lambda = float(tequila_lambda)
        self.use_hadamard = bool(use_hadamard)
        self.act_bits = int(act_bits)
        self.threshold = float(threshold)
        if not deployed_forward:
            self.norm = RMSNorm(input_dims, eps=norm_eps)

    def _extra_repr(self) -> str:
        return (
            f"input_dims={self.weight.shape[1]}, output_dims={self.weight.shape[0]}, "
            f"bias={'bias' in self}, binary={self.binary}, "
            f"deployed_forward={self.deployed_forward}, "
            f"tequila_lambda={self.tequila_lambda}, use_hadamard={self.use_hadamard}, "
            f"act_bits={self.act_bits}, threshold={self.threshold}"
        )

    def __call__(self, x: mx.array) -> mx.array:
        w = self["weight"]

        if self.binary:
            w_q = binary_weight_quant(w)
        else:
            w_q = weight_quant(w, threshold=self.threshold)
        w_quant = _quantize_ste(w, w_q)

        x_in = x if self.deployed_forward else self.norm(x)
        if self.use_hadamard:
            x_in = hadamard_wht_mx(x_in)
        if not self.deployed_forward or self.use_hadamard or self.act_bits == 4:
            act_q_fn = activation_quant_int4 if self.act_bits == 4 else activation_quant
            x_in = _quantize_ste(x_in, act_q_fn(x_in))

        if "bias" in self:
            out = mx.addmm(self["bias"], x_in, w_quant.T)
        else:
            out = x_in @ w_quant.T

        if self.tequila_lambda > 0.0 and not self.binary:
            deadzone = mx.stop_gradient((w_q == 0.0).astype(w.dtype))
            dyn_bias = self.tequila_lambda * mx.sum(w * deadzone, axis=-1)
            out = out + dyn_bias

        return out
