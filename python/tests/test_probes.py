import math

import mlx.core as mx
import pytest
from mlx.quant.probes import (
    CorticalLogMapTelemetry,
    HolographicDestructiveProbe,
    compute_snr_db,
)


def test_destructive_quantization_channel():
    shape = (64, 64)
    base_w = mx.random.normal(shape=shape)
    carrier = mx.sin(mx.linspace(0, 4 * math.pi, shape[0])[:, None]) * mx.cos(
        mx.linspace(0, 4 * math.pi, shape[1])[None, :]
    )

    probe = HolographicDestructiveProbe(carrier_amplitudes=[0.01, 0.1])
    metrics = probe.evaluate_channel(base_weights=base_w, holographic_carrier=carrier)

    assert len(metrics) == 8  # 2 amplitudes * 4 regimes

    # Group by regime for amp=0.1
    amp_01_metrics = {m.regime: m for m in metrics if m.carrier_amplitude == 0.1}

    # FP32 should have near-perfect SNR
    assert amp_01_metrics["fp32"].reconstruction_snr_db > 50.0
    # FP16 should have high SNR
    assert amp_01_metrics["fp16"].reconstruction_snr_db > 20.0
    # Phase coherence should be positive across all regimes
    for regime, m in amp_01_metrics.items():
        assert m.recovered_phase_coherence > 0.0
        assert -1.0 <= m.weight_cosine_similarity <= 1.0


def test_cortical_log_map_projection():
    shape = (32, 32)
    theta = mx.random.uniform(0.0, 2.0 * math.pi, shape=shape)
    phi = mx.random.uniform(0.0, math.pi, shape=shape)
    activations = mx.random.normal(shape=shape)

    telemetry = CorticalLogMapTelemetry()
    cx, cy = telemetry.transform_s2_to_cortical(theta, phi)

    mx.eval(cx, cy)
    assert cx.shape == shape
    assert cy.shape == shape
    assert not mx.isnan(cx).any()
    assert not mx.isnan(cy).any()

    # Planform spectrum
    spectrum = telemetry.compute_planform_spectrum(cx, cy, activations)
    assert "total_cortical_energy" in spectrum
    assert "stripes_mode_ratio" in spectrum
    assert spectrum["stripes_mode_ratio"] >= 0.0
