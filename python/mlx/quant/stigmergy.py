# Copyright © 2026 Peter Lodri / MLX-QUANT Contributors.
# Vaimshuk Step 1: Normalized Stigmergy Accumulator & Versioned Snapshot Engine.

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx


@dataclass
class StigmergySnapshot:
    """
    Immutable versioned snapshot of the gravitational density field rho.
    Supports independent replay, rollback, and ablation testing.
    """

    version: int
    timestamp: float
    shape: Tuple[int, ...]
    mean_density: float
    max_density: float
    min_density: float
    entropy: float
    total_events_processed: int
    data: Optional[mx.array] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("data", None)
        return d


class StigmergicAccumulator:
    """
    Normalized, saturating, time-smoothed transformation for swarm evidence:
        rho <- (1 - lambda) * rho + lambda * tanh(alpha * degree + beta * traversal)

    Prevents runaway positive feedback loops while giving the Armillary Sphere
    a calibrated gravitational inertia field.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        ema_decay: float = 0.05,
        alpha: float = 0.01,
        beta: float = 0.02,
        rho_min: float = 0.0,
        rho_max: float = 5.0,
        baseline_rho: float = 0.1,
    ):
        self.shape = shape
        self.ema_decay = ema_decay
        self.alpha = alpha
        self.beta = beta
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.baseline_rho = baseline_rho

        # Initialize density field with baseline
        self.density = mx.ones(shape, dtype=mx.float32) * baseline_rho
        self.event_count = 0
        self.snapshot_version = 0
        self.event_log: List[Dict[str, Any]] = []

    def ingest_event(
        self,
        indices: Optional[mx.array] = None,
        degrees: Optional[Union[float, mx.array]] = None,
        traversals: Optional[Union[float, mx.array]] = None,
        record_log: bool = False,
    ) -> mx.array:
        """
        Ingests a batch of crawler traversal evidence and applies the saturating EMA update.
        If indices is None, updates the entire field uniformly.
        """
        self.event_count += 1

        if degrees is None:
            degrees = mx.zeros(self.shape, dtype=mx.float32)
        elif not isinstance(degrees, mx.array):
            degrees = mx.ones(self.shape, dtype=mx.float32) * float(degrees)

        if traversals is None:
            traversals = mx.zeros(self.shape, dtype=mx.float32)
        elif not isinstance(traversals, mx.array):
            traversals = mx.ones(self.shape, dtype=mx.float32) * float(traversals)

        # Saturating non-linear evidence shock: tanh(alpha * deg + beta * trav)
        raw_evidence = self.alpha * degrees + self.beta * traversals
        evidence_shock = mx.tanh(raw_evidence) * (self.rho_max - self.rho_min)

        if indices is None:
            # Global smooth EMA update
            new_density = (1.0 - self.ema_decay) * self.density + self.ema_decay * (
                self.baseline_rho + evidence_shock
            )
            self.density = mx.clip(new_density, self.rho_min, self.rho_max)
        else:
            # Localized sparse update at specific tensor coordinates
            current_vals = self.density[indices]
            updated_vals = (1.0 - self.ema_decay) * current_vals + self.ema_decay * (
                self.baseline_rho + evidence_shock[indices]
            )
            self.density[indices] = mx.clip(updated_vals, self.rho_min, self.rho_max)

        if record_log:
            self.event_log.append(
                {
                    "step": self.event_count,
                    "timestamp": time.time(),
                    "has_indices": indices is not None,
                    "mean_density": float(mx.mean(self.density).item()),
                }
            )

        return self.density

    def create_snapshot(self) -> StigmergySnapshot:
        """Creates an immutable, versioned checkpoint of the current density field."""
        self.snapshot_version += 1
        d_flat = mx.reshape(self.density, (-1,))
        mean_d = float(mx.mean(d_flat).item())
        max_d = float(mx.max(d_flat).item())
        min_d = float(mx.min(d_flat).item())

        # Shannon Entropy of normalized density distribution
        probs = d_flat / (mx.sum(d_flat) + 1e-7)
        entropy = float(-mx.sum(probs * mx.log(probs + 1e-9)).item())

        return StigmergySnapshot(
            version=self.snapshot_version,
            timestamp=time.time(),
            shape=self.shape,
            mean_density=mean_d,
            max_density=max_d,
            min_density=min_d,
            entropy=entropy,
            total_events_processed=self.event_count,
            data=mx.array(self.density),
        )

    def restore_snapshot(self, snapshot: StigmergySnapshot):
        """Restores density field to a previous snapshot state."""
        if snapshot.shape != self.shape:
            raise ValueError(
                f"Snapshot shape {snapshot.shape} does not match {self.shape}"
            )
        if snapshot.data is None:
            raise ValueError("Snapshot contains no tensor data")
        self.density = mx.array(snapshot.data)
        self.snapshot_version = snapshot.version

    def reset_to_baseline(self):
        """Resets field to baseline inertia."""
        self.density = mx.ones(self.shape, dtype=mx.float32) * self.baseline_rho
        self.event_count = 0
