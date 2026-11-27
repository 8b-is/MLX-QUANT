// Pure-integer CUDA C++ compute kernels for MLX-QUANT BeeHive.
// Zero floating-point operations inside any kernel.

#if !defined(__CUDACC_RTC__)
#include <stdint.h>
#include <stddef.h>
#else
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef unsigned long size_t;
#endif

static __device__ __forceinline__ int64_t mq_saturating_mul_i64(int64_t a, int64_t b) {
    if (a == 0LL || b == 0LL) {
        return 0LL;
    }
    const int64_t i64_max = 9223372036854775807LL;
    const int64_t i64_min = (-9223372036854775807LL - 1LL);
    const bool neg = (a < 0LL) != (b < 0LL);
    const uint64_t ua = (a < 0LL) ? (uint64_t)(-(a + 1LL)) + 1ULL : (uint64_t)a;
    const uint64_t ub = (b < 0LL) ? (uint64_t)(-(b + 1LL)) + 1ULL : (uint64_t)b;
    const uint64_t max_u = neg ? ((uint64_t)i64_max + 1ULL) : (uint64_t)i64_max;
    if (ua > max_u / ub) {
        return neg ? i64_min : i64_max;
    }
    const uint64_t prod = ua * ub;
    if (neg) {
        if (prod == (uint64_t)i64_max + 1ULL) {
            return i64_min;
        }
        return -(int64_t)prod;
    }
    return (int64_t)prod;
}

static __device__ __forceinline__ int64_t mq_saturating_sub_i64(int64_t a, int64_t b) {
    const int64_t i64_max = 9223372036854775807LL;
    const int64_t i64_min = (-9223372036854775807LL - 1LL);
    if (b < 0LL && a > i64_max + b) {
        return i64_max;
    }
    if (b > 0LL && a < i64_min + b) {
        return i64_min;
    }
    return a - b;
}

static __device__ __forceinline__ int32_t mq_clamp_i64_to_i32(int64_t v) {
    if (v > 2147483647LL) {
        return 2147483647;
    }
    if (v < -2147483648LL) {
        return -2147483648;
    }
    return (int32_t)v;
}

static __device__ __forceinline__ uint64_t mq_splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// 1. 2-bit LSB-first ternary GEMV: {0, 1, 2} -> {-1, 0, +1}, code 3 -> 0.
// Row stride is cols_ceil4 = (cols + 3) / 4 bytes per row.
// Dispatch with blockDim.x as a multiple of 32 (one warp per output row).
extern "C" __global__ void ternary_2bit_gemv_i8(
    const uint8_t* __restrict__ w_packed,
    const int8_t* __restrict__ x,
    int32_t* __restrict__ y_out,
    uint32_t rows,
    uint32_t cols
) {
    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = global_tid >> 5;
    const uint32_t lane = threadIdx.x & 31u;

    if (row >= rows) {
        return;
    }

    const uint32_t cols_ceil4 = (cols + 3u) >> 2;
    const uint8_t* row_ptr = w_packed + static_cast<size_t>(row) * static_cast<size_t>(cols_ceil4);
    int32_t acc = 0;

    for (uint32_t b = lane; b < cols_ceil4; b += 32u) {
        const uint8_t packed = row_ptr[b];
        const uint32_t col_base = b << 2;
        if (col_base + 3u < cols) {
            const uint32_t c0 = packed & 0x03u;
            const uint32_t c1 = (packed >> 2) & 0x03u;
            const uint32_t c2 = (packed >> 4) & 0x03u;
            const uint32_t c3 = (packed >> 6) & 0x03u;
            const int32_t x0 = static_cast<int32_t>(x[col_base]);
            const int32_t x1 = static_cast<int32_t>(x[col_base + 1u]);
            const int32_t x2 = static_cast<int32_t>(x[col_base + 2u]);
            const int32_t x3 = static_cast<int32_t>(x[col_base + 3u]);
            acc += (c0 == 2u) ? x0 : ((c0 == 0u) ? -x0 : 0);
            acc += (c1 == 2u) ? x1 : ((c1 == 0u) ? -x1 : 0);
            acc += (c2 == 2u) ? x2 : ((c2 == 0u) ? -x2 : 0);
            acc += (c3 == 2u) ? x3 : ((c3 == 0u) ? -x3 : 0);
        } else {
            for (uint32_t k = 0u; k < 4u; ++k) {
                const uint32_t col = col_base + k;
                if (col < cols) {
                    const uint32_t code = (packed >> (k * 2u)) & 0x03u;
                    const int32_t xv = static_cast<int32_t>(x[col]);
                    acc += (code == 2u) ? xv : ((code == 0u) ? -xv : 0);
                }
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }

    if (lane == 0u) {
        y_out[row] = acc;
    }
}

// 2. BITCOS sparse ternary GEMV: presence bitmap (ceil(cols/8) bytes/row) + compacted signs.
// Dispatch with blockDim.x as a multiple of 32 (one warp per output row).
extern "C" __global__ void bitcos_gemv_i8(
    const uint8_t* __restrict__ presence,
    const uint8_t* __restrict__ signs,
    const uint32_t* __restrict__ row_sign_byte_offsets,
    const int8_t* __restrict__ x,
    int32_t* __restrict__ y_out,
    uint32_t rows,
    uint32_t cols
) {
    const uint32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = global_tid >> 5;
    const uint32_t lane = threadIdx.x & 31u;

    if (row >= rows) {
        return;
    }

    const uint32_t p_bytes = (cols + 7u) >> 3;
    const uint8_t* row_pres = presence + static_cast<size_t>(row) * static_cast<size_t>(p_bytes);
    const uint8_t* row_signs = signs + static_cast<size_t>(row_sign_byte_offsets[row]);
    const uint32_t rem_bits = cols & 7u;

    int32_t acc = 0;
    uint32_t sign_base = 0u;

    for (uint32_t base_b = 0u; base_b < p_bytes; base_b += 32u) {
        const uint32_t b = base_b + lane;
        uint32_t p_byte = 0u;
        if (b < p_bytes) {
            p_byte = static_cast<uint32_t>(row_pres[b]);
            if (b + 1u == p_bytes && rem_bits != 0u) {
                p_byte &= (1u << rem_bits) - 1u;
            }
        }

        const uint32_t cnt = static_cast<uint32_t>(__popc(p_byte));
        uint32_t prefix = cnt;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const uint32_t up = __shfl_up_sync(0xFFFFFFFFu, prefix, offset);
            if (lane >= static_cast<uint32_t>(offset)) {
                prefix += up;
            }
        }
        uint32_t s_idx = sign_base + (prefix - cnt);
        const uint32_t chunk_total = __shfl_sync(0xFFFFFFFFu, prefix, 31);
        sign_base += chunk_total;

        const uint32_t col_base = b << 3;
        while (p_byte != 0u) {
            const uint32_t bit = static_cast<uint32_t>(__ffs(static_cast<int>(p_byte)) - 1);
            p_byte &= (p_byte - 1u);
            const uint32_t col = col_base + bit;
            const int32_t xv = static_cast<int32_t>(x[col]);
            const uint32_t s_bit =
                (static_cast<uint32_t>(row_signs[s_idx >> 3]) >> (s_idx & 7u)) & 1u;
            acc += (s_bit != 0u) ? xv : -xv;
            s_idx += 1u;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }

    if (lane == 0u) {
        y_out[row] = acc;
    }
}

// 3. Tequila deadzone reactivation STE in pure 64-bit Q15.16 integer arithmetic.
// Matches mlx_quant_core::beehive::tequila_ste_step_q16 bit-for-bit.
extern "C" __global__ void tequila_deadzone_ste_q16(
    int32_t* __restrict__ weights_q16,
    const int32_t* __restrict__ grads_q16,
    uint32_t len,
    int32_t delta_q16,
    int32_t lr_q16,
    int32_t lambda_q16
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }

    const int64_t w = static_cast<int64_t>(weights_q16[idx]);
    const int64_t g = static_cast<int64_t>(grads_q16[idx]);
    const int64_t abs_w = (w < 0LL) ? -w : w;
    const int64_t delta = static_cast<int64_t>(delta_q16);
    const int64_t lr = static_cast<int64_t>(lr_q16);
    const int64_t lambda = static_cast<int64_t>(lambda_q16);

    int64_t eff_grad = g;
    if (abs_w < delta) {
        const int64_t dyn_bias = (lambda * w) >> 16;
        eff_grad += dyn_bias;
    }

    const int64_t update = mq_saturating_mul_i64(eff_grad, lr) >> 16;
    const int64_t next = mq_saturating_sub_i64(w, update);
    weights_q16[idx] = mq_clamp_i64_to_i32(next);
}

// 4. mHC 4x4 Birkhoff Sinkhorn-Knopp normalization in pure Q15.16 integer math.
extern "C" __global__ void mhc_sinkhorn_4x4_q16(
    int32_t* __restrict__ matrices_q16,
    uint32_t count,
    uint32_t iters
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }

    int32_t* mat = matrices_q16 + (static_cast<size_t>(idx) * 16u);
    uint64_t m[4][4];

    for (uint32_t r = 0u; r < 4u; ++r) {
        for (uint32_t c = 0u; c < 4u; ++c) {
            const int64_t v = static_cast<int64_t>(mat[r * 4u + c]);
            m[r][c] = (v < 1LL) ? 1ULL : static_cast<uint64_t>(v);
        }
    }

    for (uint32_t iter = 0u; iter < iters; ++iter) {
        for (uint32_t r = 0u; r < 4u; ++r) {
            const uint64_t rsum = m[r][0] + m[r][1] + m[r][2] + m[r][3];
            for (uint32_t c = 0u; c < 4u; ++c) {
                const uint64_t q = ((m[r][c] << 16) + (rsum >> 1)) / rsum;
                m[r][c] = (q < 1ULL) ? 1ULL : q;
            }
        }
        for (uint32_t c = 0u; c < 4u; ++c) {
            const uint64_t csum = m[0][c] + m[1][c] + m[2][c] + m[3][c];
            for (uint32_t r = 0u; r < 4u; ++r) {
                const uint64_t q = ((m[r][c] << 16) + (csum >> 1)) / csum;
                m[r][c] = (q < 1ULL) ? 1ULL : q;
            }
        }
    }

    for (uint32_t r = 0u; r < 4u; ++r) {
        for (uint32_t c = 0u; c < 4u; ++c) {
            mat[r * 4u + c] = mq_clamp_i64_to_i32(static_cast<int64_t>(m[r][c]));
        }
    }
}

// 5. Engram N-gram O(1) hash lookup & Q15.16 gated residual merge.
extern "C" __global__ void engram_ngram_gather_q16(
    const int32_t* __restrict__ table_q16,
    int32_t* __restrict__ hidden_q16,
    const uint32_t* __restrict__ tokens,
    uint32_t table_slots,
    uint32_t dim,
    uint32_t ngram_len,
    int32_t gate_q16
) {
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim || table_slots == 0u) {
        return;
    }

    uint64_t h = mq_splitmix64(0x9E3779B97F4A7C15ULL);
    for (uint32_t k = 0u; k < ngram_len; ++k) {
        h = mq_splitmix64(h ^ static_cast<uint64_t>(tokens[k]));
    }
    const uint32_t slot = static_cast<uint32_t>(h % static_cast<uint64_t>(table_slots));
    const int64_t gate = static_cast<int64_t>(gate_q16 < 0 ? 0 : (gate_q16 > 65536 ? 65536 : gate_q16));
    const int64_t mem_v = static_cast<int64_t>(table_q16[static_cast<size_t>(slot) * static_cast<size_t>(dim) + d]);
    const int64_t delta = (gate * mem_v) >> 16;
    hidden_q16[d] = mq_clamp_i64_to_i32(static_cast<int64_t>(hidden_q16[d]) + delta);
}
