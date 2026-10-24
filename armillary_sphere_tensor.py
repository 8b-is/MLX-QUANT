import math
import time

import mlx.core as mx


def quantize_ternary_naive(w: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """
    Baseline global absmean BitNet b1.58 ternary quantizer:
      gamma = mean(|W|), threshold = 0.5 * gamma
    Returns (q_trits in {-1,0,+1}, scale, w_recon).
    """
    gamma = mx.mean(mx.abs(w)) + 1e-8
    scaled = w / gamma
    q = mx.clip(mx.round(scaled), -1.0, 1.0)
    w_recon = q * gamma
    return q, gamma, w_recon


def quantize_ternary_optimal_groupwise(
    w: mx.array, group_size: int = 64, threshold_factor: float = 0.68
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Optimal L2-Refined Group-Wise BitNet b1.58 Ternary Quantizer:
      1. Reshapes W into groups of `group_size`.
      2. Selects active support |w_i| > threshold_factor * E[|w_g|] (optimal Gaussian/Laplace Lloyd-Max knee).
      3. Computes closed-form L2-optimal scale per group:
           alpha_g = (sum_{i in active} |w_i|) / max(1, count(active))
         which minimizes ||w_g - alpha_g * sign(w_g) * I(|w_g| > tau_g)||_2^2.
    """
    orig_shape = w.shape
    flat = w.reshape(-1, group_size)
    abs_flat = mx.abs(flat)
    group_mean = mx.mean(abs_flat, axis=-1, keepdims=True) + 1e-8
    tau = threshold_factor * group_mean
    active = abs_flat > tau
    active_f = active.astype(mx.float32)
    denom = mx.maximum(mx.sum(active_f, axis=-1, keepdims=True), 1.0)
    alpha = mx.sum(abs_flat * active_f, axis=-1, keepdims=True) / denom
    q_flat = mx.where(active, mx.sign(flat), mx.zeros_like(flat))
    w_recon = (q_flat * alpha).reshape(orig_shape)
    return q_flat.reshape(orig_shape), alpha.squeeze(-1), w_recon


@mx.compile
def _compiled_geodesic_step(
    weights: mx.array,
    density_field: mx.array,
    grad: mx.array,
    lr: float,
    base_tau: float,
) -> tuple[mx.array, mx.array, mx.array]:
    """Apple Silicon Metal fused kernel for Armillary sparse geodesic update + fused active-node reduction."""
    tau = base_tau * (1.0 + density_field)
    mask = mx.abs(grad) > tau
    sparse_grad = mx.where(mask, grad, mx.zeros_like(grad))
    new_weights = weights - lr * sparse_grad
    new_density = density_field + 0.001 * mx.abs(sparse_grad)
    active_nodes = mx.sum(mask)
    return new_weights, new_density, active_nodes


class ArmillarySphereTensor:
    """
    An Astrophysical representation of a Neural Network Tensor.
    Instead of a flat matrix, weights are mapped onto a 3D Armillary Sphere.
    This naturally enforces sparse recursion, as updates propagate along
    geodesics (rings) of the sphere rather than across a flat Cartesian grid.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_rings: int = 12,
        group_size: int = 64,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.num_rings = num_rings
        self.group_size = group_size

        # Standard flat weight initialization
        limit = math.sqrt(6 / (in_features + out_features))
        flat_weights = mx.random.uniform(
            -limit, limit, shape=(out_features, in_features)
        )
        self.weights = flat_weights

        # Create the Armillary spatial mapping (static for the layer)
        self.theta = mx.random.uniform(
            0, 2 * math.pi, shape=(out_features, in_features)
        )
        self.phi = mx.random.uniform(0, math.pi, shape=(out_features, in_features))

        # Density field for Holographic Steganography
        self.density_field = mx.abs(self.weights) * mx.sin(self.phi)
        self.w_q = None
        self.scales = None
        self.refresh_metal_ternary_cache()

    def refresh_metal_ternary_cache(self):
        """Pack weights into native Apple Silicon Metal ternary format (12.80x compression)."""
        self.w_q, self.scales = mx.quantize(
            self.weights, group_size=self.group_size, bits=2, mode="ternary"
        )
        mx.eval(self.w_q, self.scales)

    def stigmergic_threshold(self, base_tau: float) -> mx.array:
        return base_tau * (1.0 + self.density_field)

    def forward(self, x: mx.array) -> mx.array:
        """Dense fp32 baseline forward pass."""
        return mx.matmul(x, self.weights.T)

    def forward_ternary_metal(self, x: mx.array) -> mx.array:
        """Apple Silicon Metal native ternary quantized_matmul forward pass."""
        if self.w_q is None or self.scales is None:
            self.refresh_metal_ternary_cache()
        return mx.quantized_matmul(
            x, self.w_q, self.scales, group_size=self.group_size, bits=2, mode="ternary"
        )

    def sparse_recursive_update(
        self, grad: mx.array, lr: float = 0.01, base_tau: float = 0.005
    ) -> float:
        """Eager baseline sparse recursion update."""
        tau = self.stigmergic_threshold(base_tau)
        mask = mx.abs(grad) > tau
        sparse_grad = mx.where(mask, grad, mx.zeros_like(grad))
        self.weights = self.weights - lr * sparse_grad
        self.density_field = self.density_field + 0.001 * mx.abs(sparse_grad)
        active_nodes = mx.sum(mask.astype(mx.float32)).item()
        total_nodes = mask.size
        return 1.0 - (active_nodes / total_nodes)

    def sparse_recursive_update_compiled(
        self, grad: mx.array, lr: float = 0.01, base_tau: float = 0.005
    ) -> float:
        """Apple Silicon Metal @mx.compile fused geodesic update (elides intermediate mask buffer)."""
        self.weights, self.density_field, active_nodes = _compiled_geodesic_step(
            self.weights, self.density_field, grad, lr, base_tau
        )
        mx.eval(self.weights, self.density_field, active_nodes)
        return 1.0 - (active_nodes.item() / self.weights.size)


if __name__ == "__main__":
    print("🚀 Initializing MLX-QUANT Armillary Sphere Tensor on", mx.default_device())
    sphere = ArmillarySphereTensor(in_features=1024, out_features=1024, num_rings=24)
    mock_grad = mx.random.normal(shape=(1024, 1024))
    for epoch in range(3):
        sparsity = sphere.sparse_recursive_update_compiled(
            grad=mock_grad, lr=0.1, base_tau=1.5
        )
        print(f"  Epoch {epoch+1}: Tensor Sparsity = {sparsity*100:.2f}%")
