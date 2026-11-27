"""
Axiom Quant Foundation Module for MLX (`v3.0.0-alpha.1 "Ultra-OmniLLL Bare-Metal Substrate"`)
============================================================================================
Accelerated Quantitative Reasoning, Stochastic Calculus, Market Microstructure,
BitNet b1.58 / BITCOS Ternary Quantization, Exact Rational `p/q` Schedules,
DeepSeek Engram Conditional Memory, and BeeHive-ERNIE Graph/Bio Primitives.
"""

from .armillary import ArmillarySphereLinear, armillary_sparse_update_metal
from .beehive import (
    HveRlController,
    HveRlReport,
    LatentCompressedAttention,
    ManifoldHyperConnection,
    MultiMaxOutputHead,
    MushroomBodyRouter,
    ramanujan_bipartite_mask,
    sinkhorn_knopp_mx,
    spectral_gap_report,
)
from .bitnet import (
    BitcosPacked,
    TernaryLinear,
    bitcos_effective_bpw,
    bitcos_matmul_numpy,
    bitcos_pack_numpy,
    bitcos_unpack_numpy,
    convert_model_to_bitnet,
    matmul_ternary_cpu_fallback,
    quantize_ternary_numpy,
    unpack_ternary_numpy,
)
from .engram import EngramMemoryLayer, MultiHeadNgramHasher, TokenizerCompressor
from .microstructure import simulate_l2_orderbook, vpin_toxicity
from .portfolio import markowitz_efficient_frontier
from .probes import (
    CorticalLogMapTelemetry,
    DestructiveChannelMetrics,
    HolographicDestructiveProbe,
    compute_snr_db,
    quantize_to_regime,
)
from .rational import (
    ASYMMETRIC_CRITICAL_WEIGHT,
    BONOBO_RESONANCE,
    Q16_ONE,
    Q16_SHIFT,
    RATIO_ONE,
    RATIO_ZERO,
    CanaryProbeReport,
    LmHeadLrPolicy,
    Ratio,
    WsdScheduleConfig,
    compute_byte_asymmetric_mask,
    evaluate_knowledge_collapse_canaries,
)
from .shm_ring import RingRecord, SharedMemoryRingBuffer
from .stigmergy import StigmergicAccumulator, StigmergySnapshot
from .stochastic import black_scholes_greeks, simulate_gbm

__version__ = "3.0.0-alpha.1"

__all__ = [
    "__version__",
    "ArmillarySphereLinear",
    "armillary_sparse_update_metal",
    "sinkhorn_knopp_mx",
    "ManifoldHyperConnection",
    "ramanujan_bipartite_mask",
    "spectral_gap_report",
    "LatentCompressedAttention",
    "MushroomBodyRouter",
    "MultiMaxOutputHead",
    "HveRlController",
    "HveRlReport",
    "StigmergicAccumulator",
    "StigmergySnapshot",
    "SharedMemoryRingBuffer",
    "RingRecord",
    "HolographicDestructiveProbe",
    "CorticalLogMapTelemetry",
    "DestructiveChannelMetrics",
    "compute_snr_db",
    "quantize_to_regime",
    "simulate_gbm",
    "black_scholes_greeks",
    "simulate_l2_orderbook",
    "vpin_toxicity",
    "markowitz_efficient_frontier",
    "TernaryLinear",
    "BitcosPacked",
    "bitcos_pack_numpy",
    "bitcos_unpack_numpy",
    "bitcos_matmul_numpy",
    "bitcos_effective_bpw",
    "convert_model_to_bitnet",
    "quantize_ternary_numpy",
    "unpack_ternary_numpy",
    "matmul_ternary_cpu_fallback",
    "EngramMemoryLayer",
    "MultiHeadNgramHasher",
    "TokenizerCompressor",
    "Ratio",
    "RATIO_ZERO",
    "RATIO_ONE",
    "BONOBO_RESONANCE",
    "ASYMMETRIC_CRITICAL_WEIGHT",
    "Q16_ONE",
    "Q16_SHIFT",
    "WsdScheduleConfig",
    "LmHeadLrPolicy",
    "CanaryProbeReport",
    "evaluate_knowledge_collapse_canaries",
    "compute_byte_asymmetric_mask",
]
