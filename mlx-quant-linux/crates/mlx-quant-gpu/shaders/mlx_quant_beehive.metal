// Pure-integer / Q15.16 Metal compute kernels for MLX-QUANT BeeHive.
// Zero floating-point operations anywhere in any kernel.

#include <metal_stdlib>
using namespace metal;

constant int32_t Q16_ONE = 65536;
constant int64_t I32_MIN_L = -2147483648L;
constant int64_t I32_MAX_L = 2147483647L;
constant int64_t I64_MAX_L = 9223372036854775807L;
constant int64_t I64_MIN_L = (-9223372036854775807L - 1L);

static inline int32_t clamp_i64_to_i32(int64_t v) {
    if (v > I32_MAX_L) {
        return int32_t(I32_MAX_L);
    }
    if (v < I32_MIN_L) {
        return int32_t(I32_MIN_L);
    }
    return int32_t(v);
}

static inline int64_t saturating_mul_i64(int64_t a, int64_t b) {
    if (a == 0L || b == 0L) {
        return 0L;
    }
    bool neg = (a < 0L) != (b < 0L);
    uint64_t ua = (a < 0L) ? uint64_t(-(a + 1L)) + 1UL : uint64_t(a);
    uint64_t ub = (b < 0L) ? uint64_t(-(b + 1L)) + 1UL : uint64_t(b);
    uint64_t max_u = neg ? (uint64_t(I64_MAX_L) + 1UL) : uint64_t(I64_MAX_L);
    if (ua > max_u / ub) {
        return neg ? I64_MIN_L : I64_MAX_L;
    }
    uint64_t prod = ua * ub;
    if (neg) {
        if (prod == uint64_t(I64_MAX_L) + 1UL) {
            return I64_MIN_L;
        }
        return -int64_t(prod);
    }
    return int64_t(prod);
}

static inline int64_t saturating_sub_i64(int64_t a, int64_t b) {
    if (b < 0L && a > I64_MAX_L + b) {
        return I64_MAX_L;
    }
    if (b > 0L && a < I64_MIN_L + b) {
        return I64_MIN_L;
    }
    return a - b;
}

static inline uint64_t splitmix64_metal(uint64_t x) {
    x += 0x9E3779B97F4A7C15UL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
    return x ^ (x >> 31);
}

// 1. 2-bit {0, 1, 2} -> {-1, 0, +1} LSB-first ternary GEMV (code 3 -> 0).
// Row stride is cols_ceil4 = (cols + 3) / 4 bytes per row.
// Dispatched with 1 threadgroup per output row.
kernel void ternary_2bit_gemv_i8(
    device const uint8_t* w_packed [[buffer(0)]],
    device const int8_t* x [[buffer(1)]],
    device int32_t* y_out [[buffer(2)]],
    constant uint32_t& rows [[buffer(3)]],
    constant uint32_t& cols [[buffer(4)]],
    uint32_t tgid [[threadgroup_position_in_grid]],
    uint32_t tid [[thread_index_in_threadgroup]],
    uint32_t tg_size [[threads_per_threadgroup]],
    uint32_t sg_lane [[thread_index_in_simdgroup]],
    uint32_t sg_idx [[simdgroup_index_in_threadgroup]],
    uint32_t sg_count [[simdgroups_per_threadgroup]]
) {
    uint32_t row = tgid;
    if (row >= rows) {
        return;
    }

    uint32_t cols_ceil4 = (cols + 3u) >> 2;
    device const uint8_t* row_ptr = w_packed + (size_t(row) * size_t(cols_ceil4));
    int32_t acc = 0;

    for (uint32_t b = tid; b < cols_ceil4; b += tg_size) {
        uint8_t packed = row_ptr[b];
        uint32_t col_base = b << 2;
        if (col_base + 3u < cols) {
            uint32_t c0 = packed & 0x03u;
            uint32_t c1 = (packed >> 2) & 0x03u;
            uint32_t c2 = (packed >> 4) & 0x03u;
            uint32_t c3 = (packed >> 6) & 0x03u;
            int32_t x0 = int32_t(x[col_base]);
            int32_t x1 = int32_t(x[col_base + 1u]);
            int32_t x2 = int32_t(x[col_base + 2u]);
            int32_t x3 = int32_t(x[col_base + 3u]);
            acc += (c0 == 2u) ? x0 : ((c0 == 0u) ? -x0 : 0);
            acc += (c1 == 2u) ? x1 : ((c1 == 0u) ? -x1 : 0);
            acc += (c2 == 2u) ? x2 : ((c2 == 0u) ? -x2 : 0);
            acc += (c3 == 2u) ? x3 : ((c3 == 0u) ? -x3 : 0);
        } else {
            for (uint32_t k = 0u; k < 4u; ++k) {
                uint32_t col = col_base + k;
                if (col < cols) {
                    uint32_t code = (packed >> (k * 2u)) & 0x03u;
                    int32_t xv = int32_t(x[col]);
                    acc += (code == 2u) ? xv : ((code == 0u) ? -xv : 0);
                }
            }
        }
    }

    int32_t sg_total = simd_sum(acc);
    threadgroup int32_t sg_partials[64];
    if (sg_lane == 0u && sg_idx < 64u) {
        sg_partials[sg_idx] = sg_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0u) {
        uint32_t active_sgs = min(sg_count, 64u);
        int32_t partial = 0;
        for (uint32_t s = sg_lane; s < active_sgs; s += 32u) {
            partial += sg_partials[s];
        }
        int32_t row_sum = simd_sum(partial);
        if (sg_lane == 0u) {
            y_out[row] = row_sum;
        }
    }
}

// 2. BITCOS sparse ternary GEMV: presence bitmap + compacted sign bitvector.
// Dispatched with 1 SIMD group per output row.
kernel void bitcos_gemv_i8(
    device const uint8_t* presence [[buffer(0)]],
    device const uint8_t* signs [[buffer(1)]],
    device const uint32_t* row_sign_byte_offsets [[buffer(2)]],
    device const int8_t* x [[buffer(3)]],
    device int32_t* y_out [[buffer(4)]],
    constant uint32_t& rows [[buffer(5)]],
    constant uint32_t& cols [[buffer(6)]],
    uint32_t tgid [[threadgroup_position_in_grid]],
    uint32_t sg_lane [[thread_index_in_simdgroup]],
    uint32_t sg_idx [[simdgroup_index_in_threadgroup]],
    uint32_t sg_size [[threads_per_simdgroup]]
) {
    uint32_t row = tgid;
    if (row >= rows || sg_idx != 0u) {
        return;
    }

    uint32_t p_bytes = (cols + 7u) >> 3;
    device const uint8_t* row_pres = presence + (size_t(row) * size_t(p_bytes));
    device const uint8_t* row_signs = signs + size_t(row_sign_byte_offsets[row]);
    uint32_t rem_bits = cols & 7u;

    int32_t acc = 0;
    uint32_t sign_base = 0u;

    for (uint32_t base_b = 0u; base_b < p_bytes; base_b += sg_size) {
        uint32_t b = base_b + sg_lane;
        uint32_t p_byte = 0u;
        if (b < p_bytes) {
            p_byte = uint32_t(row_pres[b]);
            if (b + 1u == p_bytes && rem_bits != 0u) {
                p_byte &= (1u << rem_bits) - 1u;
            }
        }

        uint32_t cnt = popcount(p_byte);
        uint32_t s_idx = sign_base + simd_prefix_exclusive_sum(cnt);
        uint32_t chunk_total = simd_sum(cnt);
        sign_base += chunk_total;

        uint32_t col_base = b << 3;
        while (p_byte != 0u) {
            uint32_t bit = ctz(p_byte);
            p_byte &= (p_byte - 1u);
            uint32_t col = col_base + bit;
            int32_t act = int32_t(x[col]);
            uint32_t s_bit = (uint32_t(row_signs[s_idx >> 3]) >> (s_idx & 7u)) & 1u;
            acc += (s_bit != 0u) ? act : -act;
            s_idx += 1u;
        }
    }

    int32_t total = simd_sum(acc);
    if (sg_lane == 0u) {
        y_out[row] = total;
    }
}

// 3. Tequila deadzone reactivation STE in pure 64-bit Q15.16 integer arithmetic.
// Matches mlx_quant_core::beehive::tequila_ste_step_q16 bit-for-bit.
kernel void tequila_deadzone_ste_q16(
    device int32_t* weights_q16 [[buffer(0)]],
    device const int32_t* grads_q16 [[buffer(1)]],
    constant uint32_t& len [[buffer(2)]],
    constant int32_t& delta_q16 [[buffer(3)]],
    constant int32_t& lr_q16 [[buffer(4)]],
    constant int32_t& lambda_q16 [[buffer(5)]],
    uint32_t idx [[thread_position_in_grid]]
) {
    if (idx >= len) {
        return;
    }

    int64_t w = int64_t(weights_q16[idx]);
    int64_t g = int64_t(grads_q16[idx]);
    int64_t abs_w = (w < 0L) ? -w : w;
    int64_t delta = int64_t(delta_q16);
    int64_t lr = int64_t(lr_q16);
    int64_t lambda = int64_t(lambda_q16);

    int64_t eff_grad = g;
    if (abs_w < delta) {
        int64_t dyn_bias = (lambda * w) >> 16;
        eff_grad += dyn_bias;
    }

    int64_t update = saturating_mul_i64(eff_grad, lr) >> 16;
    int64_t next = saturating_sub_i64(w, update);
    weights_q16[idx] = clamp_i64_to_i32(next);
}

// 4. mHC 4x4 Birkhoff Sinkhorn-Knopp normalization in pure Q15.16 integer math.
// Each thread normalizes one 4x4 Q15.16 matrix (16 elements, row-major).
kernel void mhc_sinkhorn_4x4_q16(
    device int32_t* matrices_q16 [[buffer(0)]],
    constant uint32_t& count [[buffer(1)]],
    constant uint32_t& iters [[buffer(2)]],
    uint32_t idx [[thread_position_in_grid]]
) {
    if (idx >= count) {
        return;
    }

    device int32_t* mat = matrices_q16 + (size_t(idx) * 16u);
    uint64_t m[4][4];

    for (uint32_t r = 0u; r < 4u; ++r) {
        for (uint32_t c = 0u; c < 4u; ++c) {
            int64_t v = int64_t(mat[r * 4u + c]);
            m[r][c] = (v < 1L) ? 1UL : uint64_t(v);
        }
    }

    for (uint32_t iter = 0u; iter < iters; ++iter) {
        // Row normalization toward Q16_ONE
        for (uint32_t r = 0u; r < 4u; ++r) {
            uint64_t row_sum = m[r][0] + m[r][1] + m[r][2] + m[r][3];
            if (row_sum == 0UL) {
                row_sum = 1UL;
            }
            for (uint32_t c = 0u; c < 4u; ++c) {
                uint64_t scaled = (m[r][c] * uint64_t(Q16_ONE) + (row_sum >> 1)) / row_sum;
                m[r][c] = (scaled > 1UL) ? scaled : 1UL;
            }
        }
        // Column normalization toward Q16_ONE
        for (uint32_t c = 0u; c < 4u; ++c) {
            uint64_t col_sum = m[0][c] + m[1][c] + m[2][c] + m[3][c];
            if (col_sum == 0UL) {
                col_sum = 1UL;
            }
            for (uint32_t r = 0u; r < 4u; ++r) {
                uint64_t scaled = (m[r][c] * uint64_t(Q16_ONE) + (col_sum >> 1)) / col_sum;
                m[r][c] = (scaled > 1UL) ? scaled : 1UL;
            }
        }
    }

    for (uint32_t r = 0u; r < 4u; ++r) {
        for (uint32_t c = 0u; c < 4u; ++c) {
            uint64_t v = m[r][c];
            mat[r * 4u + c] = (v > uint64_t(I32_MAX_L)) ? int32_t(I32_MAX_L) : int32_t(v);
        }
    }
}

// 5. Engram N-gram deterministic SplitMix64 gather + Q15.16 gated residual add.
// Matches EngramTable::hash_slot + Q15.16 gated residual merge.
kernel void engram_ngram_gather_q16(
    device const uint32_t* token_ids [[buffer(0)]],
    device const int32_t* table_q16 [[buffer(1)]],
    device int32_t* out_q16 [[buffer(2)]],
    constant uint32_t& num_tokens [[buffer(3)]],
    constant uint32_t& table_slots [[buffer(4)]],
    constant uint32_t& dim [[buffer(5)]],
    constant int32_t& gate_q16 [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint32_t tok_idx = gid.y;
    uint32_t d = gid.x;
    if (tok_idx >= num_tokens || d >= dim || table_slots == 0u) {
        return;
    }

    uint64_t h = splitmix64_metal(0x9E3779B97F4A7C15UL);
    if (tok_idx > 0u) {
        h = splitmix64_metal(h ^ uint64_t(token_ids[tok_idx - 1u]));
    }
    h = splitmix64_metal(h ^ uint64_t(token_ids[tok_idx]));
    uint32_t slot = uint32_t(h % uint64_t(table_slots));

    int32_t clamped_gate = gate_q16;
    if (clamped_gate < 0) {
        clamped_gate = 0;
    } else if (clamped_gate > Q16_ONE) {
        clamped_gate = Q16_ONE;
    }

    int64_t engram_val = int64_t(table_q16[size_t(slot) * size_t(dim) + size_t(d)]);
    int64_t gated = (engram_val * int64_t(clamped_gate)) >> 16;
    size_t out_off = size_t(tok_idx) * size_t(dim) + size_t(d);
    out_q16[out_off] = clamp_i64_to_i32(int64_t(out_q16[out_off]) + gated);
}
