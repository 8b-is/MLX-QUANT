"""
DeepSeek Engram Conditional Memory for MLX (`arXiv:2601.07372`)
================================================================
O(1) deterministic multi-head N-gram hash lookup with canonical tokenizer
compression and context-aware sigmoid gating, complementing 1.58-bit ternary
neural compute layers on Apple Silicon Unified Memory.
"""

from typing import Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class TokenizerCompressor:
    """Deterministic many-to-one canonical token ID map for denser N-gram tables."""

    def __init__(
        self,
        vocab_size: int = 256,
        id_to_canonical: Optional[Sequence[int]] = None,
    ) -> None:
        self.vocab_size = vocab_size
        if id_to_canonical is not None:
            if len(id_to_canonical) != vocab_size:
                raise ValueError("id_to_canonical length must equal vocab_size")
            mapping = np.asarray(id_to_canonical, dtype=np.uint32)
        else:
            # Default byte-level canonicalization: fold ASCII uppercase -> lowercase
            # and collapse whitespace (\t, \r, \n -> space=32) for denser syntactic N-grams.
            mapping = np.arange(vocab_size, dtype=np.uint32)
            for b in range(min(vocab_size, 256)):
                if 65 <= b <= 90:  # 'A'..'Z' -> 'a'..'z'
                    mapping[b] = b + 32
                elif b in (9, 10, 13):  # tab / LF / CR -> space
                    mapping[b] = 32
        self._map_np = mapping
        self._map_mx = mx.array(mapping, dtype=mx.uint32)
        self.compressed_vocab_size = int(mapping.max()) + 1 if vocab_size > 0 else 1

    def compress(self, token_ids: mx.array) -> mx.array:
        return self._map_mx[token_ids.astype(mx.uint32)]


class MultiHeadNgramHasher:
    """O(1) multi-head modular N-gram hasher over compressed token sequences."""

    _ODD_PRIMES: Tuple[int, ...] = (
        131,
        257,
        521,
        1031,
        2053,
        4099,
        8209,
        16411,
    )

    def __init__(
        self,
        table_size: int = 4099,
        num_heads: int = 4,
        ngram_orders: Tuple[int, ...] = (2, 3),
    ) -> None:
        if table_size <= 0 or num_heads <= 0:
            raise ValueError("table_size and num_heads must be positive")
        self.table_size = table_size
        self.num_heads = num_heads
        self.ngram_orders = tuple(ngram_orders)

    def hash_sequence(self, compressed_ids: mx.array) -> mx.array:
        """
        Compute O(1) hash indices of shape `[B, T, len(ngram_orders) * num_heads]`
        with deterministic slot offsets across order and head indices.
        """
        bsz, seq_len = compressed_ids.shape
        c = compressed_ids.astype(mx.uint32)
        slot_indices = []
        slot_base = 0
        table_size_u32 = mx.array(self.table_size, dtype=mx.uint32)
        for order_idx, n in enumerate(self.ngram_orders):
            # Build shifted views c_t, c_{t-1}, ..., c_{t-n+1}
            shifts = [c]
            for lag in range(1, n):
                if seq_len <= lag:
                    shifted = mx.zeros_like(c)
                else:
                    pad = mx.zeros((bsz, lag), dtype=mx.uint32)
                    shifted = mx.concatenate([pad, c[:, : seq_len - lag]], axis=1)
                shifts.append(shifted)

            for head_idx in range(self.num_heads):
                acc = mx.full((bsz, seq_len), (order_idx + 1) * 97 + head_idx * 193, dtype=mx.uint32)
                for lag, s_arr in enumerate(shifts):
                    prime = self._ODD_PRIMES[(head_idx + lag * 3 + order_idx) % len(self._ODD_PRIMES)]
                    acc = acc + s_arr * prime
                h = (acc % table_size_u32) + slot_base
                slot_indices.append(h)
                slot_base += self.table_size

        return mx.stack(slot_indices, axis=-1)


class EngramMemoryLayer(nn.Module):
    """
    DeepSeek Engram conditional memory block (`arXiv:2601.07372`).

    Retrieves static N-gram knowledge embeddings in O(1) per position and fuses
    them into the residual stream via RMSNorm + context-aware sigmoid gating.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int = 256,
        table_size: int = 4099,
        num_heads: int = 4,
        ngram_orders: Tuple[int, ...] = (2, 3),
        head_dim: int = 16,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.table_size = table_size
        self.num_heads = num_heads
        self.ngram_orders = tuple(ngram_orders)
        self.head_dim = head_dim
        self.total_heads = len(self.ngram_orders) * num_heads
        self.total_mem_dim = self.total_heads * head_dim
        self.total_slots = self.total_heads * table_size

        self.compressor = TokenizerCompressor(vocab_size=vocab_size)
        self.hasher = MultiHeadNgramHasher(
            table_size=table_size,
            num_heads=num_heads,
            ngram_orders=self.ngram_orders,
        )

        self.table = nn.Embedding(self.total_slots, head_dim)
        # Initialize small so early residual stream isn't destabilized
        self.table.weight = mx.random.normal((self.total_slots, head_dim)) * 0.02

        self.key_proj = nn.Linear(self.total_mem_dim, d_model, bias=False)
        self.value_proj = nn.Linear(self.total_mem_dim, d_model, bias=False)
        self.query_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.key_norm = nn.RMSNorm(d_model, eps=norm_eps)

    def __call__(self, hidden: mx.array, token_ids: mx.array) -> mx.array:
        """
        Args:
            hidden: Tensor `[B, T, d_model]`
            token_ids: Integer tensor `[B, T]`
        Returns:
            Context-gated conditional memory update `[B, T, d_model]`.
        """
        compressed = self.compressor.compress(token_ids)
        indices = self.hasher.hash_sequence(compressed)  # [B, T, total_heads]
        fetched = self.table(indices)  # [B, T, total_heads, head_dim]
        e_t = fetched.reshape(hidden.shape[0], hidden.shape[1], self.total_mem_dim)

        k_t = self.key_norm(self.key_proj(e_t))
        q_t = self.query_norm(hidden)
        gate_logits = mx.sum(q_t * k_t, axis=-1, keepdims=True) * (self.d_model ** -0.5)
        alpha_t = mx.sigmoid(gate_logits)
        return alpha_t * self.value_proj(e_t)
