"""
Exact Rational (p/q) Algebra, WSD Schedules & Linger-and-Lose Collapse Guard
===========================================================================
Python mirror of OmniLLL's zero-float `Ratio { p, q }`, `WsdScheduleConfig`
(`arXiv:2404.06395`), `LmHeadLrPolicy`, and factual-capacity canary probes
(`arXiv:2609.32902`) for MLX-QUANT v2.1.0 "Zeno-Engram".
"""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

Q16_SHIFT: int = 16
Q16_ONE: int = 1 << Q16_SHIFT


class Ratio:
    """Exact non-negative canonical coprime rational `p / q` with `q >= 1`."""

    __slots__ = ("p", "q")

    def __init__(self, p: int, q: int = 1) -> None:
        if p < 0 or q < 0:
            raise ValueError(f"Ratio requires non-negative integers, got {p}/{q}")
        if p == 0 or q == 0:
            self.p = 0
            self.q = 1
            return
        if p == q:
            self.p = 1
            self.q = 1
            return
        g = math.gcd(p, q)
        self.p = p // g
        self.q = q // g

    def add_exact(self, other: "Ratio") -> "Ratio":
        return Ratio(self.p * other.q + other.p * self.q, self.q * other.q)

    def mul_exact(self, other: "Ratio") -> "Ratio":
        return Ratio(self.p * other.p, self.q * other.q)

    def div_exact(self, other: "Ratio") -> "Ratio":
        if other.p == 0:
            return RATIO_ZERO
        return Ratio(self.p * other.q, self.q * other.p)

    def sub_clamped(self, other: "Ratio") -> "Ratio":
        num = self.p * other.q - other.p * self.q
        if num <= 0:
            return RATIO_ZERO
        return Ratio(num, self.q * other.q)

    def to_q16(self) -> int:
        return (self.p << Q16_SHIFT) // self.q

    def to_float(self) -> float:
        return self.p / self.q

    def __add__(self, other: "Ratio") -> "Ratio":
        return self.add_exact(other)

    def __mul__(self, other: "Ratio") -> "Ratio":
        return self.mul_exact(other)

    def __truediv__(self, other: "Ratio") -> "Ratio":
        return self.div_exact(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ratio):
            return False
        return self.p == other.p and self.q == other.q

    def __lt__(self, other: "Ratio") -> bool:
        return self.p * other.q < other.p * self.q

    def __le__(self, other: "Ratio") -> bool:
        return self.p * other.q <= other.p * self.q

    def __gt__(self, other: "Ratio") -> bool:
        return self.p * other.q > other.p * self.q

    def __ge__(self, other: "Ratio") -> bool:
        return self.p * other.q >= other.p * self.q

    def __hash__(self) -> int:
        return hash((self.p, self.q))

    def __repr__(self) -> str:
        return f"Ratio({self.p}, {self.q})"

    def __str__(self) -> str:
        return f"{self.p}/{self.q}"


RATIO_ZERO = Ratio(0, 1)
RATIO_ONE = Ratio(1, 1)
BONOBO_RESONANCE = Ratio(3, 2)
ASYMMETRIC_CRITICAL_WEIGHT = Ratio(3, 1)

NOSTD_RUST_CRITICAL_SUBSTRINGS: tuple[str, ...] = (
    "no_std",
    "unsafe",
    "asm!",
    "Ratio",
    "Trit",
    "Q16_ONE",
    "core::",
    "repr(C)",
    "const fn",
    "#[inline",
)


@dataclass(frozen=True)
class WsdScheduleConfig:
    """Warmup-Stable-Decay LR schedule in exact rationals (`arXiv:2404.06395`)."""

    peak_lr: Ratio = Ratio(3, 10_000)
    min_lr: Ratio = Ratio(3, 100_000)
    warmup_steps: int = 100
    stable_steps: int = 700
    decay_steps: int = 200
    decay_shape: str = "one_minus_sqrt"

    def total_steps(self) -> int:
        return self.warmup_steps + self.stable_steps + self.decay_steps

    def phase_at(self, step: int) -> str:
        stable_end = self.warmup_steps + self.stable_steps
        total = stable_end + self.decay_steps
        if step < self.warmup_steps:
            return "Warmup"
        if step < stable_end:
            return "Stable"
        if step < total:
            return "Decay"
        return "Complete"

    def lr_at(self, step: int) -> Ratio:
        stable_end = self.warmup_steps + self.stable_steps
        total = stable_end + self.decay_steps
        if step < self.warmup_steps:
            if self.warmup_steps <= 0:
                return self.peak_lr
            return self.peak_lr.mul_exact(Ratio(max(0, step), self.warmup_steps))
        if step < stable_end:
            return self.peak_lr
        if step >= total or self.decay_steps <= 0:
            return self.min_lr

        dt = step - stable_end
        td = self.decay_steps
        if self.decay_shape == "linear":
            decay_frac = Ratio(td - dt, td)
        elif self.decay_shape == "quadratic":
            rem = td - dt
            decay_frac = Ratio(rem * rem, td * td)
        elif self.decay_shape == "one_minus_sqrt":
            scaled_radicand = (dt << 64) // td
            sqrt_q32 = math.isqrt(scaled_radicand)
            one_q32 = 1 << 32
            decay_frac = Ratio(max(0, one_q32 - sqrt_q32), one_q32)
        else:
            raise ValueError(f"Unknown decay_shape: {self.decay_shape}")

        if self.peak_lr <= self.min_lr:
            return self.peak_lr
        span = self.peak_lr.sub_clamped(self.min_lr)
        return self.min_lr.add_exact(span.mul_exact(decay_frac))

    def lr_float(self, step: int) -> float:
        return self.lr_at(step).to_float()


@dataclass(frozen=True)
class LmHeadLrPolicy:
    """Decoupled LM-head LR policy preventing Linger-and-Lose collapse (`arXiv:2609.32902`)."""

    body_wsd: WsdScheduleConfig = WsdScheduleConfig()
    lm_head_lr_scale: Ratio = Ratio(1, 5)
    collapse_recall_floor: Ratio = Ratio(4, 5)

    def decoupled_lm_head_lr(self, step: int) -> Ratio:
        return self.body_wsd.lr_at(step).mul_exact(self.lm_head_lr_scale)


@dataclass(frozen=True)
class CanaryProbeReport:
    total_canaries: int
    exact_recalls: int
    recall_ratio: Ratio
    bits_per_param_x1000: int
    collapse_detected: bool
    recommended_action: str


def evaluate_knowledge_collapse_canaries(
    expected_hashes: Sequence[int],
    predicted_hashes: Sequence[int],
    canary_bits_each: int,
    model_params: int,
    policy: LmHeadLrPolicy = LmHeadLrPolicy(),
) -> CanaryProbeReport:
    """Measure retained factual capacity and flag Linger-and-Lose collapse."""
    if len(expected_hashes) != len(predicted_hashes):
        raise ValueError("expected_hashes and predicted_hashes must match in length")
    total_canaries = len(expected_hashes)
    exact_recalls = sum(1 for a, b in zip(expected_hashes, predicted_hashes) if a == b)
    recall_ratio = (
        RATIO_ONE if total_canaries == 0 else Ratio(exact_recalls, total_canaries)
    )
    bits_per_param_x1000 = (
        0
        if model_params <= 0
        else (exact_recalls * canary_bits_each * 1000) // model_params
    )
    collapse_detected = total_canaries > 0 and recall_ratio < policy.collapse_recall_floor
    recommended_action = (
        "TRIGGER_WSD_COOLDOWN_AND_REDUCE_LM_HEAD_LR"
        if collapse_detected
        else "HEALTHY_CAPACITY_RETAINED"
    )
    return CanaryProbeReport(
        total_canaries=total_canaries,
        exact_recalls=exact_recalls,
        recall_ratio=recall_ratio,
        bits_per_param_x1000=bits_per_param_x1000,
        collapse_detected=collapse_detected,
        recommended_action=recommended_action,
    )


def compute_byte_asymmetric_mask(
    text: str,
    critical_weight: Ratio = ASYMMETRIC_CRITICAL_WEIGHT,
    markers: Iterable[str] = NOSTD_RUST_CRITICAL_SUBSTRINGS,
) -> list[float]:
    """Compute per-UTF8-byte loss weights (`3.0` on critical Rust `#![no_std]` spans, `1.0` elsewhere)."""
    raw = text.encode("utf-8")
    weights = [1.0] * len(raw)
    crit_val = critical_weight.to_float()
    for marker in markers:
        m_bytes = marker.encode("utf-8")
        if not m_bytes:
            continue
        start = 0
        while True:
            idx = raw.find(m_bytes, start)
            if idx < 0:
                break
            for j in range(idx, idx + len(m_bytes)):
                weights[j] = crit_val
            start = idx + 1
    return weights
