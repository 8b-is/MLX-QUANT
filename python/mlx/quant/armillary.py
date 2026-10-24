# Copyright © 2026 Peter Lodri / MLX-QUANT Contributors.
# Armillary Sphere Tensor Dynamics & Metal-Accelerated Sparse Recursion.

import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

# =====================================================================
# Metal Shading Language: Fused Armillary Sparse Recursive Kernel
# =====================================================================

METAL_ARMILLARY_SOURCE = """
    uint idx = thread_position_in_grid.x;
    if (idx >= total_elements) {
        return;
    }

    // Load inputs
    T w = weights[idx];
    T g = grad[idx];
    T d = density[idx];
    T p = phi[idx];

    // Compute astrophysical threshold tau = base_tau * (1.0 + d * abs(sin(phi)))
    T sin_phi = metal::sin(p);
    T tau = static_cast<T>(base_tau) * (static_cast<T>(1.0) + d * metal::abs(sin_phi));

    // Stigmergic Escape Velocity Condition (Sparse indicator)
    bool escapes_gravity = metal::abs(g) > tau;
    T mask_val = escapes_gravity ? static_cast<T>(1.0) : static_cast<T>(0.0);

    // Fused weight update: W_{t+1} = W_t - lr * (g * mask)
    T w_next = escapes_gravity ? (w - static_cast<T>(lr) * g) : w;

    // Fused density accumulation: rho_{t+1} = (1 - decay) * rho_t + gamma * |g| * mask
    T d_decayed = d * (static_cast<T>(1.0) - static_cast<T>(decay_rate));
    T d_next = escapes_gravity ? (d_decayed + static_cast<T>(growth_rate) * metal::abs(g)) : d_decayed;

    // Write outputs
    out_weights[idx] = w_next;
    out_density[idx] = d_next;
    out_mask[idx] = mask_val;
"""

# Global compiled Metal kernel cache
_ARMILLARY_KERNEL = None


def _get_armillary_kernel():
    global _ARMILLARY_KERNEL
    if _ARMILLARY_KERNEL is None:
        _ARMILLARY_KERNEL = mx.fast.metal_kernel(
            name="armillary_sparse_update",
            input_names=[
                "weights",
                "grad",
                "density",
                "phi",
                "lr",
                "base_tau",
                "decay_rate",
                "growth_rate",
                "total_elements",
            ],
            output_names=["out_weights", "out_density", "out_mask"],
            source=METAL_ARMILLARY_SOURCE,
        )
    return _ARMILLARY_KERNEL


def armillary_sparse_update_metal(
    weights: mx.array,
    grad: mx.array,
    density: mx.array,
    phi: mx.array,
    lr: float = 0.01,
    base_tau: float = 0.005,
    decay_rate: float = 0.001,
    growth_rate: float = 0.05,
) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Executes the fused Armillary Sparse Recursive update on Apple Silicon GPU
    via native Metal Shading Language.
    """
    kernel = _get_armillary_kernel()
    dtype = weights.dtype
    total_elements = weights.size

    # Prepare inputs
    w_flat = mx.reshape(weights, (-1,))
    g_flat = mx.reshape(grad, (-1,))
    d_flat = mx.reshape(density, (-1,))
    p_flat = mx.reshape(phi, (-1,))

    threadgroup_size = 256
    num_threadgroups = (total_elements + threadgroup_size - 1) // threadgroup_size
    grid = (num_threadgroups * threadgroup_size, 1, 1)
    tg = (threadgroup_size, 1, 1)

    outputs = kernel(
        inputs=[
            w_flat,
            g_flat,
            d_flat,
            p_flat,
            float(lr),
            float(base_tau),
            float(decay_rate),
            float(growth_rate),
            int(total_elements),
        ],
        template=[("T", dtype)],
        grid=grid,
        threadgroup=tg,
        output_shapes=[(total_elements,), (total_elements,), (total_elements,)],
        output_dtypes=[dtype, dtype, dtype],
    )

    out_weights = mx.reshape(outputs[0], weights.shape)
    out_density = mx.reshape(outputs[1], density.shape)
    out_mask = mx.reshape(outputs[2], weights.shape)

    return out_weights, out_density, out_mask


class ArmillarySphereLinear(nn.Module):
    """
    Armillary Sphere Linear Layer with Stigmergic Sparse Recursion.

    Maps weight matrices into a spherical harmonic manifold S^2 (r, theta, phi),
    where gradients propagate along celestial geodesics (rings) and updates only
    occur when gradient magnitudes overcome local gravitational field density.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        base_tau: float = 0.005,
        decay_rate: float = 0.001,
        growth_rate: float = 0.05,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.base_tau = base_tau
        self.decay_rate = decay_rate
        self.growth_rate = growth_rate

        # Orthogonal / Xavier initialization on S^2
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = mx.random.normal(shape=(out_features, in_features)) * scale

        if bias:
            self.bias = mx.zeros((out_features,))
        else:
            self.bias = None

        # Fixed celestial coordinates on S^2 for each synaptic connection
        # theta: Azimuth / Longitude [0, 2*pi)
        # phi: Colatitude / Polar angle [0, pi]
        self.theta = mx.random.uniform(
            0.0, 2.0 * math.pi, shape=(out_features, in_features)
        )
        self.phi = mx.random.uniform(0.0, math.pi, shape=(out_features, in_features))

        # Stigmergic density field (inertial mass)
        self.density = mx.abs(self.weight) * mx.sin(self.phi)

        # Holographic steganographic phase carrier
        self.holographic_phase = mx.zeros_like(self.weight)

    def __call__(self, x: mx.array) -> mx.array:
        out = mx.matmul(x, self.weight.T)
        if self.bias is not None:
            out = out + self.bias
        return out

    def update_sparse_metal(
        self, grad: mx.array, lr: float = 0.01
    ) -> Tuple[float, float]:
        """
        Executes fused Metal sparse recursion and updates layer state.
        Returns (sparsity_ratio, total_energy_dissipation).
        """
        new_weight, new_density, mask = armillary_sparse_update_metal(
            weights=self.weight,
            grad=grad,
            density=self.density,
            phi=self.phi,
            lr=lr,
            base_tau=self.base_tau,
            decay_rate=self.decay_rate,
            growth_rate=self.growth_rate,
        )

        self.weight = new_weight
        self.density = new_density

        # Steganographic Phase Encoding (Holographic Carrier modulation)
        phase_delta = mx.sin(self.theta) * mx.cos(self.phi) * mask
        self.holographic_phase = self.holographic_phase + 0.001 * phase_delta

        # Metrics
        active_count = mx.sum(mask).item()
        total_count = mask.size
        sparsity = 1.0 - (active_count / total_count)
        energy_dissipated = mx.sum(mx.abs(grad) * (1.0 - mask)).item()

        return float(sparsity), float(energy_dissipated)

    def extract_steganographic_field(self) -> mx.array:
        """
        Reconstructs the latent holographic steganographic field from the
        carrier phase matrix.
        """
        return mx.fft.rfft2(self.holographic_phase)
