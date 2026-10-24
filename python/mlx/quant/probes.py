# Copyright © 2026 Peter Lodri / MLX-QUANT Contributors.
# Vaimshuk Step 3: Destructive-Channel Holographic Probes & Cortical Telemetry.

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx


@dataclass
class DestructiveChannelMetrics:
    """Telemetry metrics from passing holographic weights through a quantization channel."""

    regime: str
    carrier_amplitude: float
    original_power: float
    error_power: float
    reconstruction_snr_db: float
    weight_cosine_similarity: float
    weight_mse: float
    recovered_phase_coherence: float


def compute_snr_db(
    signal: mx.array, reconstructed: mx.array, eps: float = 1e-10
) -> float:
    """Computes Signal-to-Noise Ratio in decibels."""
    sig_power = float(mx.sum(mx.square(signal)).item())
    noise_power = float(mx.sum(mx.square(signal - reconstructed)).item())
    if noise_power < eps:
        return 100.0  # Cap perfect reconstruction
    return 10.0 * math.log10((sig_power + eps) / (noise_power + eps))


def quantize_to_regime(w: mx.array, regime: str) -> mx.array:
    """Simulates destructive quantization channels."""
    if regime == "fp32":
        return w.astype(mx.float32)
    elif regime == "fp16":
        return w.astype(mx.float16).astype(mx.float32)
    elif regime == "int8":
        # Symmetric 8-bit scale
        scale = mx.max(mx.abs(w)) / 127.0
        scale = mx.maximum(scale, 1e-7)
        q = mx.round(w / scale)
        q = mx.clip(q, -128.0, 127.0)
        return (q * scale).astype(mx.float32)
    elif regime == "bitnet_b158":
        # BitNet b1.58 ternary {-1, 0, +1}
        scale = mx.mean(mx.abs(w))
        scale = mx.maximum(scale, 1e-7)
        w_scaled = w / scale
        q = mx.round(w_scaled)
        q = mx.clip(q, -1.0, 1.0)
        return (q * scale).astype(mx.float32)
    else:
        raise ValueError(f"Unknown quantization regime: {regime}")


class HolographicDestructiveProbe:
    """
    Experimental probe that treats ternary quantization as a noisy transmission channel
    and measures the empirical recoverability of the Fourier-Bessel phase carrier.
    """

    def __init__(self, carrier_amplitudes: Optional[List[float]] = None):
        self.carrier_amplitudes = carrier_amplitudes or [0.001, 0.005, 0.01, 0.05, 0.1]
        self.regimes = ["fp32", "fp16", "int8", "bitnet_b158"]

    def evaluate_channel(
        self,
        base_weights: mx.array,
        holographic_carrier: mx.array,
    ) -> List[DestructiveChannelMetrics]:
        """
        Injects carrier into base weights across amplitudes, passes through quantization channels,
        and measures signal degradation and phase recovery.
        """
        results: List[DestructiveChannelMetrics] = []

        for amp in self.carrier_amplitudes:
            # Modulate weights with carrier
            modulated_w = base_weights + amp * holographic_carrier

            for regime in self.regimes:
                # Pass through destructive quantization channel
                degraded_w = quantize_to_regime(modulated_w, regime)

                # Attempt to extract carrier residual
                extracted_carrier = (degraded_w - base_weights) / amp

                # Compute Metrics
                snr = compute_snr_db(holographic_carrier, extracted_carrier)

                # Weight MSE & Cosine Similarity
                w_mse = float(mx.mean(mx.square(modulated_w - degraded_w)).item())
                w_norm_prod = float(
                    (
                        mx.sqrt(mx.sum(mx.square(modulated_w)))
                        * mx.sqrt(mx.sum(mx.square(degraded_w)))
                    ).item()
                )
                cos_sim = max(
                    -1.0,
                    min(
                        1.0,
                        float(mx.sum(modulated_w * degraded_w).item())
                        / (w_norm_prod + 1e-9),
                    ),
                )

                # Phase coherence (normalized dot product in Fourier space)
                f_orig = mx.fft.rfft2(holographic_carrier)
                f_recv = mx.fft.rfft2(extracted_carrier)
                orig_abs = mx.abs(f_orig)
                recv_abs = mx.abs(f_recv)
                coherence = float(mx.sum(orig_abs * recv_abs).item()) / (
                    float(
                        mx.sqrt(
                            mx.sum(mx.square(orig_abs)) * mx.sum(mx.square(recv_abs))
                        ).item()
                    )
                    + 1e-9
                )

                results.append(
                    DestructiveChannelMetrics(
                        regime=regime,
                        carrier_amplitude=amp,
                        original_power=float(
                            mx.mean(mx.square(holographic_carrier)).item()
                        ),
                        error_power=float(
                            mx.mean(
                                mx.square(holographic_carrier - extracted_carrier)
                            ).item()
                        ),
                        reconstruction_snr_db=snr,
                        weight_cosine_similarity=cos_sim,
                        weight_mse=w_mse,
                        recovered_phase_coherence=coherence,
                    )
                )

        return results


class CorticalLogMapTelemetry:
    """
    Downstream visualization & telemetry transform based on the Schwartz
    Retino-Cortical complex-log map:
        x = (alpha / eps) * ln(1 + (eps / w0) * r)
        y = beta * r * theta / (w0 + eps * r)
    """

    def __init__(
        self,
        w0: float = 0.087,
        eps: float = 0.051,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        self.w0 = w0
        self.eps = eps
        self.alpha = alpha
        self.beta = beta

    def transform_s2_to_cortical(
        self, theta: mx.array, phi: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Projects S^2 spherical angles (colatitude phi as radial r) into cortical V1 coordinates.
        Pure observer transform downstream of compute kernel.
        """
        r = phi  # [0, pi]
        # Drasdo/Schwartz continuous mapping
        x = (self.alpha / self.eps) * mx.log(1.0 + (self.eps / self.w0) * r)
        y = (self.beta * r * theta) / (self.w0 + self.eps * r)
        return x, y

    def compute_planform_spectrum(
        self, cortical_x: mx.array, cortical_y: mx.array, activations: mx.array
    ) -> Dict[str, float]:
        """
        Computes Klüver mode spectrum from cortical activations (stripes vs spirals vs lattices).
        """
        fft_act = mx.fft.rfft2(activations)
        mag = mx.abs(fft_act)
        total_energy = float(mx.sum(mag).item()) + 1e-9

        # Low frequency / fundamental wavenumber modes
        h, w = mag.shape
        stripes_energy = float(mx.sum(mag[1:4, 0]).item()) / total_energy
        lattice_energy = float(mx.sum(mag[1:4, 1:4]).item()) / total_energy
        cobweb_energy = float(mx.sum(mag[0, 1:4]).item()) / total_energy

        return {
            "total_cortical_energy": total_energy,
            "stripes_mode_ratio": stripes_energy,
            "lattice_mode_ratio": lattice_energy,
            "cobweb_mode_ratio": cobweb_energy,
        }
