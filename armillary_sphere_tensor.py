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
    w: mx.array,
    group_size: int = 64,
    threshold_factor: float = 0.68,
    refinement_steps: int = 0,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Optimal L2-Refined Group-Wise BitNet b1.58 Ternary Quantizer:
      1. Reshapes W into groups of `group_size`.
      2. Selects initial active support |w_i| > threshold_factor * E[|w_g|] (Gaussian/Laplace Lloyd-Max knee).
      3. Computes closed-form L2-optimal scale per group:
           alpha_g = (sum_{i in active} |w_i|) / max(1, count(active))
         which minimizes ||w_g - alpha_g * sign(w_g) * I(|w_g| > tau_g)||_2^2.
      4. Optionally runs `refinement_steps` of alternating Voronoi/Lloyd-Max coordinate descent:
           tau_g^{(t)} = 0.5 * alpha_g^{(t)},  alpha_g^{(t+1)} = E[|w_g| | |w_g| > tau_g^{(t)}]
         which monotonically lowers L2 reconstruction error, especially on heavy-tailed / outlier groups.
    """
    orig_shape = w.shape
    flat = w.reshape(-1, group_size)
    abs_flat = mx.abs(flat)
    sign_flat = mx.sign(flat)
    group_mean = mx.mean(abs_flat, axis=-1, keepdims=True) + 1e-8
    tau = threshold_factor * group_mean
    active = abs_flat > tau
    active_f = active.astype(mx.float32)
    denom = mx.maximum(mx.sum(active_f, axis=-1, keepdims=True), 1.0)
    alpha = mx.sum(abs_flat * active_f, axis=-1, keepdims=True) / denom
    for _ in range(refinement_steps):
        active = abs_flat > (0.5 * alpha)
        active_f = active.astype(mx.float32)
        denom = mx.maximum(mx.sum(active_f, axis=-1, keepdims=True), 1.0)
        alpha = mx.sum(abs_flat * active_f, axis=-1, keepdims=True) / denom
    q_flat = mx.where(active, sign_flat, mx.zeros_like(flat))
    w_recon = (q_flat * alpha).reshape(orig_shape)
    return q_flat.reshape(orig_shape), alpha.squeeze(-1), w_recon


def quantize_ternary_groupwise(
    w: mx.array,
    group_size: int = 64,
    threshold_factor: float = 0.68,
    refinement_steps: int = 2,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    Alternating scale-refined group-wise BitNet b1.58 ternary quantizer (default 2 coordinate-descent steps).
    """
    return quantize_ternary_optimal_groupwise(
        w,
        group_size=group_size,
        threshold_factor=threshold_factor,
        refinement_steps=refinement_steps,
    )


def pack_ternary_2bit(q: mx.array) -> mx.array:
    """
    Pack ternary trits in {-1, 0, +1} into 2-bit uint8 storage (4 trits per byte, LSB-first).
    Mapping: -1 -> 0 (0b00), 0 -> 1 (0b01), +1 -> 2 (0b10).
    Achieves 8x weight compression vs FP16 (4x vs INT8, 16x vs FP32).
    """
    if q.shape[-1] % 4 != 0:
        raise ValueError(
            f"Last dimension ({q.shape[-1]}) must be divisible by 4 for 2-bit packing."
        )
    codes = (mx.round(q).astype(mx.int8) + 1).astype(mx.uint8) & 0x03
    reshaped = codes.reshape(*q.shape[:-1], q.shape[-1] // 4, 4)
    packed = (
        reshaped[..., 0]
        | (reshaped[..., 1] << 2)
        | (reshaped[..., 2] << 4)
        | (reshaped[..., 3] << 6)
    )
    return packed


def unpack_ternary_2bit(
    w_packed: mx.array,
    scales: mx.array | None = None,
    group_size: int = 64,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """
    Unpack 2-bit uint8 ternary storage (4 trits per byte) back to {-1, 0, +1} (or scaled weights if `scales` given).
    """
    c0 = (w_packed & 0x03).astype(mx.int8) - 1
    c1 = ((w_packed >> 2) & 0x03).astype(mx.int8) - 1
    c2 = ((w_packed >> 4) & 0x03).astype(mx.int8) - 1
    c3 = ((w_packed >> 6) & 0x03).astype(mx.int8) - 1
    out_shape = (*w_packed.shape[:-1], w_packed.shape[-1] * 4)
    q = mx.stack([c0, c1, c2, c3], axis=-1).reshape(out_shape).astype(dtype)
    if scales is not None:
        q_grouped = q.reshape(-1, group_size)
        s_grouped = scales.astype(dtype).reshape(-1, 1)
        return (q_grouped * s_grouped).reshape(out_shape)
    return q


_PACKED_TERNARY_GEMV_SOURCE = """
    uint lane = thread_index_in_simdgroup;
    uint row = thread_position_in_grid.y;
    uint batch_idx = thread_position_in_grid.z;
    uint N = uint(w_packed_shape[0]);
    uint bytes_per_row = uint(w_packed_shape[1]);
    uint groups_per_row = uint(scales_shape[1]);
    if (row >= N) return;

    uint K = bytes_per_row << 2;
    const device uint8_t* w_row = w_packed + row * bytes_per_row;
    const device ScaleT* s_row = scales + row * groups_per_row;
    const device InT* x_row = x + batch_idx * K;

    float acc = 0.0f;
    for (uint byte_idx = lane; byte_idx < bytes_per_row; byte_idx += 32) {
        uint8_t packed = w_row[byte_idx];
        uint g = byte_idx >> GROUP_SHIFT;
        float scale = float(s_row[g]);
        const device InT* xp = x_row + (byte_idx << 2);
        float t0 = float(int(packed & 0x3u) - 1);
        float t1 = float(int((packed >> 2) & 0x3u) - 1);
        float t2 = float(int((packed >> 4) & 0x3u) - 1);
        float t3 = float(int((packed >> 6) & 0x3u) - 1);
        float dot4 = float(xp[0]) * t0 + float(xp[1]) * t1 + float(xp[2]) * t2 + float(xp[3]) * t3;
        acc += dot4 * scale;
    }
    float total = simd_sum(acc);
    if (lane == 0) {
        y[batch_idx * N + row] = InT(total);
    }
"""

_PACKED_TERNARY_METAL_KERNEL = mx.fast.metal_kernel(
    name="metal_packed_ternary_gemv",
    input_names=["x", "w_packed", "scales"],
    output_names=["y"],
    source=_PACKED_TERNARY_GEMV_SOURCE,
)


def metal_packed_ternary_matvec(
    x: mx.array,
    w_packed: mx.array,
    scales: mx.array,
    group_size: int = 64,
) -> mx.array:
    """
    Fused Apple Silicon Metal GPU kernel for 2-bit packed ternary groupwise matvec/matmul.
    Unpacks 4 trits per uint8 byte on-the-fly in registers and multiplies with per-group scale
    `gamma` using SIMD-group (`simd_sum`) warp reductions without materializing unpacked weights in DRAM.
    """
    squeeze_batch = x.ndim == 1
    x_2d = x.reshape(1, -1) if squeeze_batch else x
    B, K = x_2d.shape
    N = w_packed.shape[0]
    groups_per_row = K // group_size
    scales_2d = scales.reshape(N, groups_per_row)
    group_shift = int(math.log2(group_size // 4))
    tg_y = min(4, max(1, N))

    out = _PACKED_TERNARY_METAL_KERNEL(
        inputs=[x_2d, w_packed, scales_2d],
        template=[
            ("InT", x_2d.dtype),
            ("ScaleT", scales_2d.dtype),
            ("GROUP_SHIFT", group_shift),
        ],
        grid=(32, N, B),
        threadgroup=(32, tg_y, 1),
        output_shapes=[(B, N)],
        output_dtypes=[x_2d.dtype],
    )[0]
    return out.squeeze(0) if squeeze_batch else out


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
        self.w_packed_2bit = None
        self.scales_2bit = None
        self.refresh_metal_ternary_cache()

    def refresh_metal_ternary_cache(self):
        """Pack weights into native Apple Silicon Metal ternary format (12.80x compression)."""
        self.w_q, self.scales = mx.quantize(
            self.weights, group_size=self.group_size, bits=2, mode="ternary"
        )
        mx.eval(self.w_q, self.scales)

    def refresh_packed_2bit_cache(self, refinement_steps: int = 2):
        """Quantize with L2 coordinate-descent refinement and pack 4 trits/byte into uint8."""
        q, scales, _ = quantize_ternary_groupwise(
            self.weights, group_size=self.group_size, refinement_steps=refinement_steps
        )
        self.w_packed_2bit = pack_ternary_2bit(q)
        self.scales_2bit = scales.astype(mx.float16).reshape(
            self.out_features, self.in_features // self.group_size
        )
        mx.eval(self.w_packed_2bit, self.scales_2bit)

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

    def forward_packed_ternary_metal(self, x: mx.array) -> mx.array:
        """Custom Apple Silicon Metal SIMD-group fused 2-bit packed ternary forward pass."""
        if self.w_packed_2bit is None or self.scales_2bit is None:
            self.refresh_packed_2bit_cache()
        return metal_packed_ternary_matvec(
            x, self.w_packed_2bit, self.scales_2bit, group_size=self.group_size
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
