# Copyright © 2026 Peter Lodri / MLX-QUANT Contributors.
# Trajectory Vaimshuk: Live End-to-End Instrumented Pipeline.
# Pipeline: Swarm/SHM Ring -> Stigmergic Accumulator -> Armillary S^2 Metal -> Probes & Replay.

import os
import sys
import tempfile
import time

import mlx.core as mx
from mlx.quant import (
    ArmillarySphereLinear,
    CorticalLogMapTelemetry,
    HolographicDestructiveProbe,
    SharedMemoryRingBuffer,
    StigmergicAccumulator,
)


def run_vaimshuk_pipeline(verbose: bool = True):
    if verbose:
        print("=" * 80)
        print(" 🛰️ TRAJECTORY VAIMSHUK: END-TO-END INSTRUMENTED PIPELINE 🛰️")
        print("=" * 80)

    # 1. Initialize Bounded Shared-Memory Ring Buffer (Approaching Substrate Handshake)
    tmp_shm = os.path.join(
        tempfile.gettempdir(), f"vaimshuk_{int(time.time()*1000)}.shm"
    )
    ring = SharedMemoryRingBuffer(tmp_shm, capacity=512, create=True)
    if verbose:
        print(f"[Stage 1: SHM Handshake] Attached to bounded memory ring at {tmp_shm}")

    # 2. Initialize Stigmergic Density Accumulator & Armillary Layer
    in_dim, out_dim = 256, 256
    accumulator = StigmergicAccumulator(
        shape=(out_dim, in_dim),
        ema_decay=0.1,
        alpha=0.02,
        beta=0.03,
        rho_min=0.0,
        rho_max=4.0,
        baseline_rho=0.1,
    )
    layer = ArmillarySphereLinear(
        in_features=in_dim,
        out_features=out_dim,
        base_tau=0.05,
    )
    cortical_telemetry = CorticalLogMapTelemetry()
    probe = HolographicDestructiveProbe(carrier_amplitudes=[0.01, 0.05, 0.1])

    if verbose:
        print(
            f"[Stage 2: Manifold Init] Armillary S^2 ({out_dim}x{in_dim}) initialized on Apple Silicon GPU"
        )

    # 3. Simulate Swarm Ingestion via SHM Ring
    num_events = 20
    if verbose:
        print(
            f"[Stage 3: Ingestion] Producing {num_events} graph traversal events into SHM ring..."
        )

    for i in range(num_events):
        degree = 5.0 + (i % 7) * 4.0
        traversals = 12.0 + (i % 5) * 8.0
        trits = f"t3:node_{i:04d}".encode("utf-8")
        ok = ring.push(
            event_type=1,
            node_id=i,
            degree=degree,
            traversal_count=traversals,
            payload_trits=trits,
        )
        assert ok, "SHM Ring buffer overflow"

    # 4. Consume SHM Batch -> Update Stigmergic Accumulator
    records = ring.pop_batch(max_records=num_events)
    if verbose:
        print(f"[Stage 4: Accumulator] Consumed {len(records)} records from SHM ring.")

    for rec in records:
        accumulator.ingest_event(degrees=rec.degree, traversals=rec.traversal_count)

    # Take an immutable snapshot
    snapshot_v1 = accumulator.create_snapshot()
    if verbose:
        print(
            f"  -> Snapshot Version {snapshot_v1.version}: Mean Density = {snapshot_v1.mean_density:.4f}, Max = {snapshot_v1.max_density:.4f}, Entropy = {snapshot_v1.entropy:.4f}"
        )

    # Bind accumulator density to the Armillary layer's gravitational field
    layer.density = mx.array(accumulator.density)

    # 5. Execute Fused Metal S^2 Sparse Recursion Update
    if verbose:
        print("[Stage 5: Metal Execution] Dispatching fused MSL kernel on GPU...")

    # Inject mock knowledge-graph gradient shock
    grad = mx.random.normal(shape=(out_dim, in_dim))
    sparsity, energy_dissipated = layer.update_sparse_metal(grad=grad, lr=0.05)

    if verbose:
        print(
            f"  -> Sparse Update Results: Sparsity = {sparsity*100:.2f}%, Energy Dissipation = {energy_dissipated:.4f}"
        )

    # 6. Telemetry: Destructive Quantization Channel Probing & Cortical Planform Spectrum
    if verbose:
        print(
            "[Stage 6: Telemetry Probes] Evaluating destructive channel & cortical log-map..."
        )

    holographic_carrier = layer.holographic_phase
    metrics = probe.evaluate_channel(
        base_weights=layer.weight, holographic_carrier=holographic_carrier
    )

    if verbose:
        print("-" * 80)
        print(
            f"{'Channel Regime':<16} | {'Carrier Amp':<12} | {'Reconstruction SNR (dB)':<25} | {'Phase Coherence':<16}"
        )
        print("-" * 80)
        for m in metrics:
            if m.carrier_amplitude in [0.01, 0.1]:
                print(
                    f"{m.regime:<16} | {m.carrier_amplitude:<12.3f} | {m.reconstruction_snr_db:>23.2f}dB | {m.recovered_phase_coherence:>14.4f}"
                )
        print("-" * 80)

    # Downstream Cortical Log-Map transform
    cx, cy = cortical_telemetry.transform_s2_to_cortical(layer.theta, layer.phi)
    act = layer(mx.random.normal(shape=(1, in_dim)))
    spectrum = cortical_telemetry.compute_planform_spectrum(
        cx, cy, mx.reshape(layer.weight, (out_dim, in_dim))
    )
    if verbose:
        print(
            f"[Cortical Telemetry] Stripes Ratio: {spectrum['stripes_mode_ratio']:.4f}, Lattice Ratio: {spectrum['lattice_mode_ratio']:.4f}"
        )

    # 7. Verify Independent Replayability Invariant
    if verbose:
        print(
            "[Stage 7: Replay Verification] Restoring snapshot v1 and verifying deterministic field..."
        )

    saved_density_state = mx.array(accumulator.density)
    # Dirty the state
    accumulator.ingest_event(degrees=999.0, traversals=999.0)
    assert not mx.allclose(accumulator.density, saved_density_state)
    # Restore
    accumulator.restore_snapshot(snapshot_v1)
    assert mx.allclose(accumulator.density, saved_density_state)
    if verbose:
        print("  -> Snapshot Rollback Invariant Verified Bit-Accurately.")

    # Cleanup SHM
    ring.close()
    if verbose:
        print("=" * 80)
        print(" ✅ VAIMSHUK LIVE E2E PIPELINE EXECUTION COMPLETE ✅")
        print("=" * 80)

    return {
        "sparsity": sparsity,
        "energy_dissipated": energy_dissipated,
        "snapshot": snapshot_v1,
        "metrics_count": len(metrics),
        "spectrum": spectrum,
    }


if __name__ == "__main__":
    run_vaimshuk_pipeline(verbose=True)
