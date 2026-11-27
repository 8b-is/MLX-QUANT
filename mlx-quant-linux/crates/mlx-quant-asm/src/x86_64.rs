//! x86_64 AVX2, AVX-512BW/VNNI, L1 prefetch, and RDTSC hardware counter kernels.

use core::arch::x86_64::{
    _MM_HINT_T0, __m128i, __m256i, _mm_add_epi32, _mm_cvtsi128_si32, _mm_hadd_epi32,
    _mm_loadu_si128, _mm_prefetch, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_cvtepi8_epi16,
    _mm256_extracti128_si256, _mm256_madd_epi16, _mm256_max_epi16, _mm256_min_epi16,
    _mm256_set1_epi16, _mm512_add_epi8, _mm512_dpbusd_epi32, _mm512_loadu_si512, _mm512_max_epi8,
    _mm512_min_epi8, _mm512_reduce_add_epi32, _mm512_set1_epi8, _mm512_setzero_si512,
    _mm512_sub_epi32, _rdtsc,
};

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

/// 64-lane balanced-ternary `i8` dot product using x86_64 AVX2 intrinsics.
///
/// Widens inputs to `i16` via `_mm256_cvtepi8_epi16`, clamps weights to `{-1, 0, +1}`,
/// and multiplies via `_mm256_madd_epi16` so `w = -1, a = i8::MIN (-128)` yields `+128`
/// without `VPSIGNB` overflow.
///
/// # Safety
/// Caller must guarantee the CPU supports `avx2` (checked via `is_x86_feature_detected!("avx2")`
/// or compile-time `target_feature = "avx2"`).
#[target_feature(enable = "avx2")]
pub unsafe fn ternary_dot_64_avx2(w: &[i8; 64], a: &[i8; 64]) -> i32 {
    // SAFETY: Caller upholds `avx2` target feature; `w` and `a` are 64-byte slices read
    // in four unaligned 16-byte chunks each.
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
