// Pure-integer CUDA C++ compute kernels for MLX-QUANT BeeHive ternary & BITCOS GEMV.
// Zero floating-point operations.

#if !defined(__CUDACC_RTC__)
#include <stdint.h>
#else
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned int uint32_t;
typedef int int32_t;
#endif

// 2-bit LSB-first ternary GEMV: {0, 1, 2} -> {-1, 0, +1}, code 3 -> 0.
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

// BITCOS sparse ternary GEMV: presence bitmap (ceil(cols/8) bytes/row) + compacted signs.
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
