//! x86_64 AVX2, AVX-512BW/VNNI, L1 prefetch, and RDTSC hardware counter kernels.

use core::arch::x86_64::{
    _MM_HINT_T0, __m128i, __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_hadd_epi32,
    _mm_loadl_epi64, _mm_loadu_si128, _mm_prefetch, _mm256_add_epi32, _mm256_and_si256,
    _mm256_castsi256_si128, _mm256_cvtepi8_epi16, _mm256_cvtepu8_epi32, _mm256_extracti128_si256,
    _mm256_madd_epi16, _mm256_max_epi16, _mm256_min_epi16, _mm256_or_si256, _mm256_set1_epi16,
    _mm256_set1_epi32, _mm256_setr_epi8, _mm256_shuffle_epi8, _mm256_slli_epi32,
    _mm512_add_epi8, _mm512_and_si512, _mm512_cmpeq_epi8_mask, _mm512_cvtepu8_epi32,
    _mm512_dpbusd_epi32, _mm512_loadu_si512, _mm512_mask_blend_epi8, _mm512_max_epi8,
    _mm512_min_epi8, _mm512_or_si512, _mm512_reduce_add_epi32, _mm512_set1_epi8,
    _mm512_set1_epi32, _mm512_setzero_si512, _mm512_slli_epi32, _mm512_sub_epi32, _rdtsc,
};
#[cfg(target_feature = "avx2")]
use core::arch::x86_64::{_mm256_slli_epi16, _mm256_srai_epi16};

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_epi32_avx2(v: __m256i) -> i32 {
    let lo: __m128i = _mm256_castsi256_si128(v);
    let hi: __m128i = _mm256_extracti128_si256::<1>(v);
    let sum128 = _mm_add_epi32(lo, hi);
    let sum64 = _mm_hadd_epi32(sum128, sum128);
    let sum32 = _mm_hadd_epi32(sum64, sum64);
    _mm_cvtsi128_si32(sum32)
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn unpack_2bit_32_avx2(p8: __m128i) -> __m256i {
    // Expand 8 packed bytes -> 8 u32 lanes, broadcast each 2-bit pair across 4 bytes:
    // byte0=b&3, byte1=(b>>2)&3, byte2=(b>>4)&3, byte3=(b>>6)&3.
    let v = _mm256_cvtepu8_epi32(p8);
    let t1 = _mm256_slli_epi32::<6>(v);
    let t2 = _mm256_slli_epi32::<12>(v);
    let t3 = _mm256_slli_epi32::<18>(v);
    let packed_u = _mm256_or_si256(_mm256_or_si256(v, t1), _mm256_or_si256(t2, t3));
    let codes = _mm256_and_si256(packed_u, _mm256_set1_epi32(0x0303_0303));
    // LUT mapping {0, 1, 2, 3} -> {-1, 0, +1, 0} in signed i8 across both 128-bit lanes.
    let lut = _mm256_setr_epi8(
        -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0,
    );
    _mm256_shuffle_epi8(lut, codes)
}

/// 64-lane balanced-ternary `i8` dot product using x86_64 AVX2 intrinsics.
///
/// Widens inputs to `i16` via `_mm256_cvtepi8_epi16`, clamps weights to `{-1, 0, +1}`,
/// and multiplies via `_mm256_madd_epi16` so `w = -1, a = i8::MIN (-128)` yields `+128`
/// without `VPSIGNB` overflow.
///
/// # Safety
/// Caller must guarantee the CPU supports `avx2`.
#[target_feature(enable = "avx2")]
pub unsafe fn ternary_dot_64_avx2(w: &[i8; 64], a: &[i8; 64]) -> i32 {
    // SAFETY: Caller guarantees `avx2`; `w` and `a` are 64-byte slices read in four 16-byte chunks.
    unsafe {
        let ones = _mm256_set1_epi16(1);
        let neg_ones = _mm256_set1_epi16(-1);

        let w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().cast()));
        let w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(16).cast()));
        let w2 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(32).cast()));
        let w3 = _mm256_cvtepi8_epi16(_mm_loadu_si128(w.as_ptr().add(48).cast()));

        let a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().cast()));
        let a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(16).cast()));
        let a2 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(32).cast()));
        let a3 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(48).cast()));

        let cw0 = _mm256_max_epi16(_mm256_min_epi16(w0, ones), neg_ones);
        let cw1 = _mm256_max_epi16(_mm256_min_epi16(w1, ones), neg_ones);
        let cw2 = _mm256_max_epi16(_mm256_min_epi16(w2, ones), neg_ones);
        let cw3 = _mm256_max_epi16(_mm256_min_epi16(w3, ones), neg_ones);

        let p0 = _mm256_madd_epi16(cw0, a0);
        let p1 = _mm256_madd_epi16(cw1, a1);
        let p2 = _mm256_madd_epi16(cw2, a2);
        let p3 = _mm256_madd_epi16(cw3, a3);

        let acc = _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3));
        hsum_epi32_avx2(acc)
    }
}

/// 64-lane 2-bit packed ternary dot product (`{0,1,2} -> {-1,0,+1}`, `3 -> 0`) using AVX2.
///
/// # Safety
/// Caller must guarantee the CPU supports `avx2`.
#[target_feature(enable = "avx2")]
pub unsafe fn ternary_2bit_dot_64_avx2(packed: &[u8; 16], a: &[i8; 64]) -> i32 {
    // SAFETY: Caller guarantees `avx2`; `packed` is 16 bytes and `a` is 64 bytes.
    unsafe {
        let p_lo = _mm_loadl_epi64(packed.as_ptr().cast());
        let p_hi = _mm_loadl_epi64(packed.as_ptr().add(8).cast());

        let w_lo = unpack_2bit_32_avx2(p_lo);
        let w_hi = unpack_2bit_32_avx2(p_hi);

        let w0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_lo));
        let w1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(w_lo));
        let w2 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_hi));
        let w3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(w_hi));

        let a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().cast()));
        let a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(16).cast()));
        let a2 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(32).cast()));
        let a3 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(48).cast()));

        let p0 = _mm256_madd_epi16(w0, a0);
        let p1 = _mm256_madd_epi16(w1, a1);
        let p2 = _mm256_madd_epi16(w2, a2);
        let p3 = _mm256_madd_epi16(w3, a3);

        let acc = _mm256_add_epi32(_mm256_add_epi32(p0, p1), _mm256_add_epi32(p2, p3));
        hsum_epi32_avx2(acc)
    }
}

/// 64-lane balanced-ternary `i8` dot product using AVX-512BW + AVX-512VNNI.
///
/// Shifts clamped ternary weights into unsigned `u = w_clamped + 1` in `{0, 1, 2}` and
/// computes `dpbusd(0, u, a) - dpbusd(0, 1, a)`, completely avoiding signed byte negation
/// overflow when `a[i] == i8::MIN (-128)`.
///
/// # Safety
/// Caller must guarantee the CPU supports `avx512f`, `avx512bw`, and `avx512vnni`.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn ternary_dot_64_avx512(w: &[i8; 64], a: &[i8; 64]) -> i32 {
    // SAFETY: Caller upholds `avx512f`, `avx512bw`, `avx512vnni`; `w` and `a` are 64 bytes each.
    unsafe {
        let wv = _mm512_loadu_si512(w.as_ptr().cast());
        let av = _mm512_loadu_si512(a.as_ptr().cast());

        let ones = _mm512_set1_epi8(1);
        let neg_ones = _mm512_set1_epi8(-1);
        let w_clamped = _mm512_max_epi8(_mm512_min_epi8(wv, ones), neg_ones);
        let u = _mm512_add_epi8(w_clamped, ones);

        let zero = _mm512_setzero_si512();
        let sum_ua = _mm512_dpbusd_epi32(zero, u, av);
        let sum_a = _mm512_dpbusd_epi32(zero, ones, av);
        let diff = _mm512_sub_epi32(sum_ua, sum_a);
        _mm512_reduce_add_epi32(diff)
    }
}

/// 64-lane 2-bit packed ternary dot product (`{0,1,2} -> {-1,0,+1}`, `3 -> 0`) using
/// AVX-512BW + AVX-512VNNI (`_mm512_dpbusd_epi32`).
///
/// # Safety
/// Caller must guarantee the CPU supports `avx512f`, `avx512bw`, and `avx512vnni`.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub unsafe fn ternary_2bit_dot_64_avx512(packed: &[u8; 16], a: &[i8; 64]) -> i32 {
    // SAFETY: Caller guarantees `avx512f`, `avx512bw`, `avx512vnni`; `packed` is 16 bytes.
    unsafe {
        let p128 = _mm_loadu_si128(packed.as_ptr().cast());
        let v = _mm512_cvtepu8_epi32(p128);
        let t1 = _mm512_slli_epi32::<6>(v);
        let t2 = _mm512_slli_epi32::<12>(v);
        let t3 = _mm512_slli_epi32::<18>(v);
        let packed_u = _mm512_or_si512(_mm512_or_si512(v, t1), _mm512_or_si512(t2, t3));
        let codes = _mm512_and_si512(packed_u, _mm512_set1_epi32(0x0303_0303));

        // Map sentinel `3` -> `1` so unsigned `u ∈ {0, 1, 2}` evaluates `u - 1 ∈ {-1, 0, +1}`.
        let ones = _mm512_set1_epi8(1);
        let threes = _mm512_set1_epi8(3);
        let mask3 = _mm512_cmpeq_epi8_mask(codes, threes);
        let u = _mm512_mask_blend_epi8(mask3, codes, ones);

        let av = _mm512_loadu_si512(a.as_ptr().cast());
        let zero = _mm512_setzero_si512();
        let sum_ua = _mm512_dpbusd_epi32(zero, u, av);
        let sum_a = _mm512_dpbusd_epi32(zero, ones, av);
        _mm512_reduce_add_epi32(_mm512_sub_epi32(sum_ua, sum_a))
    }
}

/// Exact arbitrary-length `x86_64` 2-bit packed ternary GEMV (`y_out = W * x`).
pub fn ternary_2bit_gemv_x86(
    w_packed: &[u8],
    x: &[i8],
    y_out: &mut [i32],
    rows: usize,
    cols: usize,
) {
    let row_bytes = cols.div_ceil(4);
    assert!(
        w_packed.len() >= rows.saturating_mul(row_bytes),
        "w_packed buffer too small"
    );
    assert!(x.len() >= cols, "x buffer too small");
    assert!(y_out.len() >= rows, "y_out buffer too small");

    if rows == 0 {
        return;
    }
    if cols == 0 {
        y_out[..rows].fill(0);
        return;
    }

    let chunks64 = cols >> 6;
    let rem_cols = cols & 63;

    for r in 0..rows {
        let row_ptr = &w_packed[r * row_bytes..(r + 1) * row_bytes];
        if r + 1 < rows {
            prefetch_l1_x86(w_packed[(r + 1) * row_bytes..].as_ptr());
        }
        let mut acc = 0i32;
        for blk in 0..chunks64 {
            let byte_off = blk << 4;
            let col_off = blk << 6;
            let mut p16 = [0u8; 16];
            p16.copy_from_slice(&row_ptr[byte_off..byte_off + 16]);
            let mut a64 = [0i8; 64];
            a64.copy_from_slice(&x[col_off..col_off + 64]);

            #[cfg(all(
                target_feature = "avx512f",
                target_feature = "avx512bw",
                target_feature = "avx512vnni"
            ))]
            {
                // SAFETY: Compile-time `target_feature` guarantees AVX-512F/BW/VNNI.
                acc += unsafe { ternary_2bit_dot_64_avx512(&p16, &a64) };
            }
            #[cfg(all(
                target_feature = "avx2",
                not(all(
                    target_feature = "avx512f",
                    target_feature = "avx512bw",
                    target_feature = "avx512vnni"
                ))
            ))]
            {
                // SAFETY: Compile-time `target_feature` guarantees AVX2.
                acc += unsafe { ternary_2bit_dot_64_avx2(&p16, &a64) };
            }
            #[cfg(not(target_feature = "avx2"))]
            {
                acc += mlx_quant_core::ternary::ternary_2bit_dot_64_scalar(&p16, &a64);
            }
        }
        if rem_cols != 0 {
            let tail_byte_off = chunks64 << 4;
            let tail_col_off = chunks64 << 6;
            acc += mlx_quant_core::ternary::ternary_2bit_dot_scalar(
                &row_ptr[tail_byte_off..],
                &x[tail_col_off..],
                rem_cols,
            );
        }
        y_out[r] = acc;
    }
}

/// 32-element (16-byte) signed-nibble INT4 dot product chunk using AVX2 (`_mm256_madd_epi16`).
///
/// # Safety
/// Caller must guarantee the CPU supports `avx2` and `a_ptr`/`b_ptr` point to `>= 16` readable bytes.
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn int4_dot_32_avx2(a_ptr: *const u8, b_ptr: *const u8) -> i32 {
    // SAFETY: Caller guarantees `avx2` and valid 16-byte reads from `a_ptr` and `b_ptr`.
    unsafe {
        let a128 = _mm_loadu_si128(a_ptr.cast());
        let b128 = _mm_loadu_si128(b_ptr.cast());
        let a16 = _mm256_cvtepi8_epi16(a128);
        let b16 = _mm256_cvtepi8_epi16(b128);

        let a_lo = _mm256_srai_epi16::<12>(_mm256_slli_epi16::<12>(a16));
        let b_lo = _mm256_srai_epi16::<12>(_mm256_slli_epi16::<12>(b16));
        let a_hi = _mm256_srai_epi16::<12>(_mm256_slli_epi16::<8>(a16));
        let b_hi = _mm256_srai_epi16::<12>(_mm256_slli_epi16::<8>(b16));

        let acc = _mm256_add_epi32(
            _mm256_madd_epi16(a_lo, b_lo),
            _mm256_madd_epi16(a_hi, b_hi),
        );
        hsum_epi32_avx2(acc)
    }
}

/// Exact `x86_64` INT4 signed-nibble GEMM: `C[m x n] = A[m x k] * B^T[n x k]`.
pub fn int4_gemm_x86(
    a_packed: &[u8],
    b_t_packed: &[u8],
    c_out: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
) {
    #[cfg(target_feature = "avx2")]
    {
        let k_bytes = k.div_ceil(2);
        assert!(
            a_packed.len() >= m.saturating_mul(k_bytes),
            "a_packed buffer too small"
        );
        assert!(
            b_t_packed.len() >= n.saturating_mul(k_bytes),
            "b_t_packed buffer too small"
        );
        assert!(c_out.len() >= m.saturating_mul(n), "c_out buffer too small");
        if m == 0 || n == 0 {
            return;
        }
        if k == 0 {
            c_out[..m * n].fill(0);
            return;
        }

        let full_chunks32 = k >> 5;
        let full_chunk_bytes = full_chunks32 << 4;
        let full_tail_bytes = (k >> 1) - full_chunk_bytes;
        let has_nibble_tail = (k & 1) != 0;

        for r in 0..m {
            let a_row = &a_packed[r * k_bytes..(r + 1) * k_bytes];
            for col in 0..n {
                let b_row = &b_t_packed[col * k_bytes..(col + 1) * k_bytes];
                let mut acc = 0i32;
                for c_idx in 0..full_chunks32 {
                    let off = c_idx << 4;
                    // SAFETY: `off + 16 <= k_bytes`, compile-time `target_feature = "avx2"`.
                    acc += unsafe {
                        int4_dot_32_avx2(a_row[off..].as_ptr(), b_row[off..].as_ptr())
                    };
                }
                for b_idx in full_chunk_bytes..(full_chunk_bytes + full_tail_bytes) {
                    let (a0, a1) = mlx_quant_core::ternary::int4_unpack_byte(a_row[b_idx]);
                    let (b0, b1) = mlx_quant_core::ternary::int4_unpack_byte(b_row[b_idx]);
                    acc += (a0 as i32) * (b0 as i32) + (a1 as i32) * (b1 as i32);
                }
                if has_nibble_tail {
                    let tail_b = k >> 1;
                    let (a0, _) = mlx_quant_core::ternary::int4_unpack_byte(a_row[tail_b]);
                    let (b0, _) = mlx_quant_core::ternary::int4_unpack_byte(b_row[tail_b]);
                    acc += (a0 as i32) * (b0 as i32);
                }
                c_out[r * n + col] = acc;
            }
        }
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        mlx_quant_core::ternary::int4_gemm_scalar(a_packed, b_t_packed, c_out, m, n, k);
    }
}

/// Issue a non-faulting L1 temporal data cache prefetch (`_MM_HINT_T0`) on x86_64.
#[inline]
pub fn prefetch_l1_x86(ptr: *const u8) {
    if !ptr.is_null() {
        // SAFETY: `_mm_prefetch` is a non-faulting architectural cache hint on x86_64.
        unsafe {
            _mm_prefetch::<{ _MM_HINT_T0 }>(ptr.cast::<i8>());
        }
    }
}

/// Read the x86_64 Time Stamp Counter (`RDTSC`).
#[inline]
pub fn read_rdtsc() -> u64 {
    // SAFETY: `RDTSC` is an unprivileged architectural instruction on x86_64.
    unsafe { _rdtsc() }
}
