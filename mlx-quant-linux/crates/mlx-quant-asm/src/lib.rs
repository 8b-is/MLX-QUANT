#![cfg_attr(not(test), no_std)]
//! Cross-architecture bare-metal SIMD / inline-assembly compute kernels for `mlx-quant`.
//!
//! Provides AArch64 NEON / `FEAT_DotProd` inline-assembly kernels, x86_64 AVX2 /
//! AVX-512BW+VNNI kernels, Apple Silicon AMX tile plans and `.inst` opcode definitions,
//! hardware cycle counters, L1 prefetch hints, and clean cross-architecture dispatchers.

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
pub mod amx;
#[cfg(target_arch = "x86_64")]
pub mod x86_64;

pub use amx::{
    AMX_CLR, AMX_LDX, AMX_LDY, AMX_MAC16, AMX_SET, AMX_STZ, AmxTilePlan, amx_backend_status,
};

/// Cross-architecture 64-lane balanced-ternary `i8` dot product.
#[inline]
pub fn ternary_dot_64(weights: &[i8; 64], activations: &[i8; 64]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_feature = "dotprod")]
        {
            aarch64::ternary_dot_64_neon_dotprod(weights, activations)
        }
        #[cfg(not(target_feature = "dotprod"))]
        {
            aarch64::ternary_dot_64_neon_baseline(weights, activations)
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(all(
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vnni"
        ))]
        {
            // SAFETY: Compile-time `target_feature` guarantees AVX-512F/BW/VNNI support.
            unsafe { x86_64::ternary_dot_64_avx512(weights, activations) }
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
            // SAFETY: Compile-time `target_feature` guarantees AVX2 support.
            unsafe { x86_64::ternary_dot_64_avx2(weights, activations) }
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            mlx_quant_core::ternary::ternary_dot_64_scalar(weights, activations)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        mlx_quant_core::ternary::ternary_dot_64_scalar(weights, activations)
    }
}

/// Cross-architecture 64-lane 2-bit packed ternary dot product (`{0,1,2} -> {-1,0,+1}`).
#[inline]
pub fn ternary_2bit_dot_64(packed: &[u8; 16], activations: &[i8; 64]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::ternary_2bit_dot_64_neon(packed, activations)
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(all(
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vnni"
        ))]
        {
            // SAFETY: Compile-time `target_feature` guarantees AVX-512F/BW/VNNI support.
            unsafe { x86_64::ternary_2bit_dot_64_avx512(packed, activations) }
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
            // SAFETY: Compile-time `target_feature` guarantees AVX2 support.
            unsafe { x86_64::ternary_2bit_dot_64_avx2(packed, activations) }
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            mlx_quant_core::ternary::ternary_2bit_dot_64_scalar(packed, activations)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        mlx_quant_core::ternary::ternary_2bit_dot_64_scalar(packed, activations)
    }
}

/// Cross-architecture 2-bit packed ternary GEMV (`y_out[..rows] = W * x`).
#[inline]
pub fn ternary_2bit_gemv(
    w_packed: &[u8],
    x: &[i8],
    y_out: &mut [i32],
    rows: usize,
    cols: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::ternary_2bit_gemv_neon(w_packed, x, y_out, rows, cols);
    }
    #[cfg(target_arch = "x86_64")]
    {
        x86_64::ternary_2bit_gemv_x86(w_packed, x, y_out, rows, cols);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        mlx_quant_core::ternary::ternary_2bit_gemv_scalar(w_packed, x, y_out, rows, cols);
    }
}

/// Cross-architecture **Any-Rate 2-Plane Progressive Residual Ternary GEMV** (`arXiv:2611.34402`)
/// in zero-float Q15.16 fixed-point (`active_planes = 1` or `2`).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn anyrate_2plane_gemv(
    w_plane0: &[u8],
    w_plane1: &[u8],
    scales0_q16: &[i32],
    scales1_q16: &[i32],
    x: &[i8],
    y_out_q16: &mut [i64],
    rows: usize,
    cols: usize,
    active_planes: u8,
) {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::anyrate_2plane_gemv_neon(
            w_plane0,
            w_plane1,
            scales0_q16,
            scales1_q16,
            x,
            y_out_q16,
            rows,
            cols,
            active_planes,
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        assert!(y_out_q16.len() >= rows);
        let row_bytes = cols.div_ceil(4);
        let use_p1 = active_planes >= 2;
        for r in 0..rows {
            let row0 = &w_plane0[r * row_bytes..(r + 1) * row_bytes];
            let row1 = &w_plane1[r * row_bytes..(r + 1) * row_bytes];
            let mut acc0 = 0i64;
            let mut acc1 = 0i64;
            for c in 0..cols {
                let shift = (c & 3) << 1;
                let xi = x[c] as i64;
                acc0 += match (row0[c >> 2] >> shift) & 0x03 {
                    0 => -xi,
                    2 => xi,
                    _ => 0,
                };
                if use_p1 {
                    acc1 += match (row1[c >> 2] >> shift) & 0x03 {
                        0 => -xi,
                        2 => xi,
                        _ => 0,
                    };
                }
            }
            let mut total = (acc0 as i128) * (scales0_q16[r] as i128);
            if use_p1 {
                total += (acc1 as i128) * (scales1_q16[r] as i128);
            }
            y_out_q16[r] = (total >> 16) as i64;
        }
    }
}

/// Cross-architecture INT4 signed-nibble GEMM (`C = A * B^T`).
#[inline]
pub fn int4_gemm(
    a_packed: &[u8],
    b_t_packed: &[u8],
    c_out: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::int4_gemm_neon(a_packed, b_t_packed, c_out, m, n, k);
    }
    #[cfg(target_arch = "x86_64")]
    {
        x86_64::int4_gemm_x86(a_packed, b_t_packed, c_out, m, n, k);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        mlx_quant_core::ternary::int4_gemm_scalar(a_packed, b_t_packed, c_out, m, n, k);
    }
}

/// Cross-architecture BITCOS sparse presence+sign GEMV with L1 cache prefetching.
pub fn bitcos_gemv(
    presence: &[u8],
    signs: &[u8],
    row_sign_byte_offsets: &[u32],
    x: &[i8],
    y_out: &mut [i32],
    rows: usize,
    cols: usize,
) {
    assert!(y_out.len() >= rows, "y_out buffer too small");
    assert!(x.len() >= cols, "x activation slice too small");
    assert!(
        row_sign_byte_offsets.len() >= rows,
        "row_sign_byte_offsets slice too small"
    );
    let p_row_bytes = cols.div_ceil(8);
    assert!(
        presence.len() >= rows.saturating_mul(p_row_bytes),
        "presence buffer too small"
    );

    if rows == 0 {
        return;
    }
    if cols == 0 {
        y_out[..rows].fill(0);
        return;
    }

    let mut r = 0usize;
    while r < rows {
        if r + 1 < rows {
            let next_p = (r + 1) * p_row_bytes;
            if next_p < presence.len() {
                prefetch_l1(presence[next_p..].as_ptr());
            }
            let next_s = row_sign_byte_offsets[r + 1] as usize;
            if next_s < signs.len() {
                prefetch_l1(signs[next_s..].as_ptr());
            }
        }
        let p_slice = &presence[r * p_row_bytes..(r + 1) * p_row_bytes];
        let s_start = row_sign_byte_offsets[r] as usize;
        assert!(s_start <= signs.len(), "row_sign_byte_offsets out of bounds");
        let s_slice = &signs[s_start..];
        y_out[r] = mlx_quant_core::ternary::bitcos_dot_i8(cols, p_slice, s_slice, &x[..cols]);
        r += 1;
    }
}

/// Cross-architecture in-place 16-element saturating Walsh-Hadamard Transform (`H-BitLinear`).
#[inline]
pub fn hadamard_wht_16(block: &mut [i32; 16]) {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::hadamard_wht_16_neon(block);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = mlx_quant_core::ternary::hadamard_wht_inplace_i32(block);
    }
}

/// Issue a non-faulting L1 temporal data prefetch hint for `ptr` (no-op if null or unsupported).
#[inline]
pub fn prefetch_l1(ptr: *const u8) {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::prefetch_l1_aarch64(ptr);
    }
    #[cfg(target_arch = "x86_64")]
    {
        x86_64::prefetch_l1_x86(ptr);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = ptr;
    }
}

/// Read the architecture's unprivileged hardware counter (`cntvct_el0` on AArch64, `rdtsc` on x86_64).
#[inline]
pub fn hardware_counter() -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::read_cntvct_el0()
    }
    #[cfg(target_arch = "x86_64")]
    {
        x86_64::read_rdtsc()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        0
    }
}

/// Return the hardware counter frequency in Hz when architecturally exposed (`cntfrq_el0` on AArch64).
#[inline]
pub fn hardware_counter_freq_hz() -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        aarch64::read_cntfrq_el0()
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        0
    }
}

/// Return the active ISA backend identifier compiled into this target.
#[inline]
pub const fn active_isa_name() -> &'static str {
    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(target_feature = "dotprod")]
        {
            "aarch64-neon-dotprod"
        }
        #[cfg(not(target_feature = "dotprod"))]
        {
            "aarch64-neon-baseline"
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(all(
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vnni"
        ))]
        {
            "x86_64-avx512vnni"
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
            "x86_64-avx2"
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            "x86_64-scalar"
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        "scalar"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_quant_core::ratio::splitmix64;
    use mlx_quant_core::ternary::{
        Trit, bitcos_gemv_scalar, bitcos_pack_into, hadamard_wht_inplace_i32, int4_gemm_scalar,
        ternary_2bit_dot_64_scalar, ternary_2bit_gemv_scalar, ternary_dot_64_scalar,
    };

    #[test]
    fn amx_plan_and_status_are_sane() {
        let plan = AmxTilePlan::default();
        assert_eq!(plan.tile_m, 32);
        assert_eq!(plan.tile_n, 32);
        assert_eq!(plan.tile_k, 32);
        assert_eq!(AMX_SET, 0x0020_1220);
        assert_eq!(AMX_CLR, 0x0020_1221);
        assert_eq!(AMX_LDX, 0x0020_1000);
        assert_eq!(AMX_LDY, 0x0020_1020);
        assert_eq!(AMX_STZ, 0x0020_10A0);
        assert_eq!(AMX_MAC16, 0x0020_11C0);
        #[cfg(not(feature = "apple-amx"))]
        assert_eq!(
            amx_backend_status(),
            "designed/opt-in (enable feature apple-amx)"
        );
        assert!(!active_isa_name().is_empty());
    }

    #[test]
    fn ternary_dot_64_edge_and_fuzz_parity() {
        let edge_patterns: [([i8; 64], [i8; 64]); 8] = [
            ([0i8; 64], [0i8; 64]),
            ([i8::MIN; 64], [i8::MIN; 64]),
            ([i8::MAX; 64], [i8::MAX; 64]),
            ([i8::MIN; 64], [i8::MAX; 64]),
            ([1i8; 64], [i8::MIN; 64]),
            ([-1i8; 64], [i8::MIN; 64]),
            ([2i8; 64], [i8::MIN; 64]),
            ([-2i8; 64], [i8::MAX; 64]),
        ];
        for (w, a) in &edge_patterns {
            let expected = ternary_dot_64_scalar(w, a);
            assert_eq!(ternary_dot_64(w, a), expected);
            #[cfg(target_arch = "aarch64")]
            assert_eq!(aarch64::ternary_dot_64_neon_baseline(w, a), expected);
            #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
            assert_eq!(aarch64::ternary_dot_64_neon_dotprod(w, a), expected);
        }

        let mut seed = 0x517C_C1B7_2722_0A95u64;
        for iter in 0..10_000u32 {
            let mut w = [0i8; 64];
            let mut a = [0i8; 64];
            for idx in 0..64 {
                seed = splitmix64(seed ^ ((idx as u64) << 12) ^ u64::from(iter));
                w[idx] = if (iter & 1) == 0 {
                    match seed % 3 {
                        0 => -1,
                        1 => 0,
                        _ => 1,
                    }
                } else {
                    (seed & 0xFF) as i8
                };
                a[idx] = ((seed >> 8) & 0xFF) as i8;
            }
            let expected = ternary_dot_64_scalar(&w, &a);
            assert_eq!(ternary_dot_64(&w, &a), expected, "dispatch iter={iter}");
            #[cfg(target_arch = "aarch64")]
            assert_eq!(
                aarch64::ternary_dot_64_neon_baseline(&w, &a),
                expected,
                "baseline iter={iter}"
            );
            #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
            assert_eq!(
                aarch64::ternary_dot_64_neon_dotprod(&w, &a),
                expected,
                "dotprod iter={iter}"
            );
        }
    }

    #[test]
    fn ternary_2bit_dot_64_edge_and_fuzz_parity() {
        let edge_patterns: [([u8; 16], [i8; 64]); 7] = [
            ([0x00u8; 16], [i8::MIN; 64]), // all -1 * -128
            ([0x55u8; 16], [i8::MIN; 64]), // all 0 * -128
            ([0xAAu8; 16], [i8::MIN; 64]), // all +1 * -128
            ([0xFFu8; 16], [i8::MIN; 64]), // all code 3 -> 0 * -128
            ([0xFFu8; 16], [i8::MAX; 64]), // all code 3 -> 0 * 127
            ([0x00u8; 16], [i8::MAX; 64]), // all -1 * 127
            ([0xE4u8; 16], [i8::MIN; 64]), // repeating [0,1,2,3] -> [-1,0,+1,0]
        ];
        for (p, a) in &edge_patterns {
            let expected = ternary_2bit_dot_64_scalar(p, a);
            assert_eq!(ternary_2bit_dot_64(p, a), expected);
            #[cfg(target_arch = "aarch64")]
            assert_eq!(aarch64::ternary_2bit_dot_64_neon_baseline(p, a), expected);
            #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
            assert_eq!(aarch64::ternary_2bit_dot_64_neon_dotprod(p, a), expected);
        }

        let mut seed = 0xD1B5_4A32_D192_ED03u64;
        for iter in 0..10_000u32 {
            let mut packed = [0u8; 16];
            let mut a = [0i8; 64];
            if (iter & 1) == 0 {
                // Canonical {0, 1, 2} 2-bit weights packed LSB-first.
                for b in &mut packed {
                    let mut byte = 0u8;
                    for slot in 0..4 {
                        seed = splitmix64(seed);
                        let code = (seed % 3) as u8;
                        byte |= code << (slot * 2);
                    }
                    *b = byte;
                }
            } else {
                // Full 0..=255 range including code 3 -> 0.
                for b in &mut packed {
                    seed = splitmix64(seed);
                    *b = (seed & 0xFF) as u8;
                }
            }
            for elem in &mut a {
                seed = splitmix64(seed);
                *elem = ((seed >> 16) & 0xFF) as i8;
            }

            let expected = ternary_2bit_dot_64_scalar(&packed, &a);
            assert_eq!(
                ternary_2bit_dot_64(&packed, &a),
                expected,
                "2bit dispatch iter={iter}"
            );
            #[cfg(target_arch = "aarch64")]
            assert_eq!(
                aarch64::ternary_2bit_dot_64_neon_baseline(&packed, &a),
                expected,
                "2bit baseline iter={iter}"
            );
            #[cfg(all(target_arch = "aarch64", target_feature = "dotprod"))]
            assert_eq!(
                aarch64::ternary_2bit_dot_64_neon_dotprod(&packed, &a),
                expected,
                "2bit dotprod iter={iter}"
            );
        }
    }

    #[test]
    fn ternary_2bit_gemv_odd_tails_and_fuzz_parity() {
        let shapes = [(0usize, 0usize), (3, 0), (1, 1), (17, 73), (9, 131), (4, 128)];
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for iter in 0..64u32 {
            for &(rows, cols) in &shapes {
                let row_bytes = cols.div_ceil(4);
                let mut w_packed = vec![0u8; rows * row_bytes];
                let mut x = vec![0i8; cols];
                for b in &mut w_packed {
                    seed = splitmix64(seed);
                    *b = (seed & 0xFF) as u8;
                }
                for xi in &mut x {
                    seed = splitmix64(seed);
                    *xi = ((seed >> 8) & 0xFF) as i8;
                }
                if cols > 0 {
                    x[0] = i8::MIN;
                    x[cols - 1] = i8::MAX;
                }

                let mut expected = vec![0i32; rows];
                let mut actual = vec![-1i32; rows];
                ternary_2bit_gemv_scalar(&w_packed, &x, &mut expected, rows, cols);
                ternary_2bit_gemv(&w_packed, &x, &mut actual, rows, cols);
                assert_eq!(actual, expected, "iter={iter} rows={rows} cols={cols}");
            }
        }
    }

    #[test]
    fn int4_gemm_tiled_and_odd_tail_parity() {
        let pack_nibble = |lo: i8, hi: i8| -> u8 { ((lo as u8) & 0x0F) | (((hi as u8) & 0x0F) << 4) };

        // Extreme nibble boundary test on 32-element dot chunk
        let a_ext = [pack_nibble(-8, 7); 16];
        let b_ext = [pack_nibble(-8, -7); 16];
        let mut c_ext_ref = [0i32; 1];
        let mut c_ext_got = [0i32; 1];
        int4_gemm_scalar(&a_ext, &b_ext, &mut c_ext_ref, 1, 1, 32);
        int4_gemm(&a_ext, &b_ext, &mut c_ext_got, 1, 1, 32);
        assert_eq!(c_ext_got, c_ext_ref);
        #[cfg(target_arch = "aarch64")]
        assert_eq!(aarch64::int4_dot_32_neon_baseline(&a_ext, &b_ext), c_ext_ref[0]);

        let shapes = [
            (0usize, 0usize, 0usize),
            (3, 4, 0),
            (1, 1, 1),
            (5, 7, 38),
            (5, 7, 37),
            (6, 9, 65),
            (8, 12, 96),
            (17, 13, 73),
        ];
        let mut seed = 0xBF58_476D_1CE4_E5B9u64;
        for iter in 0..64u32 {
            for &(m, n, k) in &shapes {
                let packed_k = k.div_ceil(2);
                let mut a_packed = vec![0u8; m * packed_k];
                let mut b_t_packed = vec![0u8; n * packed_k];
                for b in a_packed.iter_mut().chain(b_t_packed.iter_mut()) {
                    seed = splitmix64(seed);
                    *b = if (iter & 1) == 0 {
                        let lo = ((seed % 15) as i8) - 7;
                        let hi = (((seed >> 8) % 15) as i8) - 7;
                        pack_nibble(lo, hi)
                    } else {
                        // Full [-8..=7] signed nibble coverage
                        (seed & 0xFF) as u8
                    };
                }
                let mut expected = vec![0i32; m * n];
                let mut actual = vec![i32::MIN; m * n];
                int4_gemm_scalar(&a_packed, &b_t_packed, &mut expected, m, n, k);
                int4_gemm(&a_packed, &b_t_packed, &mut actual, m, n, k);
                assert_eq!(actual, expected, "iter={iter} m={m} n={n} k={k}");
            }
        }
    }

    #[test]
    fn bitcos_gemv_parity() {
        let rows = 17usize;
        let cols = 73usize;
        let p_row_bytes = cols.div_ceil(8);
        let mut presence = vec![0u8; rows * p_row_bytes];
        let mut signs = Vec::new();
        let mut offsets = Vec::with_capacity(rows);
        let mut x = vec![0i8; cols];

        let mut seed = 0x94D0_49BB_1331_11EBu64;
        for xi in &mut x {
            seed = splitmix64(seed);
            *xi = (seed & 0xFF) as i8;
        }
        x[0] = i8::MIN;
        x[cols - 1] = i8::MAX;

        for r in 0..rows {
            offsets.push(signs.len() as u32);
            let mut trits = vec![Trit::Zero; cols];
            if r != 5 {
                // Leave row 5 all-zero to test empty-sign offset boundary.
                for t in &mut trits {
                    seed = splitmix64(seed);
                    *t = match seed % 4 {
                        0 => Trit::Neg,
                        1 => Trit::Pos,
                        _ => Trit::Zero,
                    };
                }
            }
            let mut s_row = vec![0u8; cols.div_ceil(8)];
            let (_nnz, s_used) = bitcos_pack_into(
                &trits,
                &mut presence[r * p_row_bytes..(r + 1) * p_row_bytes],
                &mut s_row,
            );
            signs.extend_from_slice(&s_row[..s_used]);
        }

        let mut expected = vec![0i32; rows];
        let mut actual = vec![0i32; rows];
        bitcos_gemv_scalar(&presence, &signs, &offsets, &x, &mut expected, rows, cols);
        bitcos_gemv(&presence, &signs, &offsets, &x, &mut actual, rows, cols);
        assert_eq!(actual, expected);
    }

    #[test]
    fn hadamard_wht_16_edge_and_fuzz_parity() {
        let edge_blocks: [[i32; 16]; 4] = [
            [0; 16],
            [i32::MAX; 16],
            [i32::MIN; 16],
            [
                i32::MAX,
                i32::MIN,
                i32::MAX,
                i32::MIN,
                1,
                -1,
                i32::MAX / 2,
                i32::MIN / 2,
                0,
                123_456,
                -654_321,
                i32::MAX,
                i32::MIN,
                7,
                -7,
                42,
            ],
        ];
        for blk in &edge_blocks {
            let mut expected = *blk;
            let mut actual = *blk;
            assert!(hadamard_wht_inplace_i32(&mut expected));
            hadamard_wht_16(&mut actual);
            assert_eq!(actual, expected);
        }

        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        for iter in 0..10_000u32 {
            let mut blk = [0i32; 16];
            for v in &mut blk {
                seed = splitmix64(seed);
                *v = if (iter & 1) == 0 {
                    (seed & 0xFFFF_FFFF) as i32
                } else {
                    ((seed & 0xFFFF) as i16) as i32
                };
            }
            let mut expected = blk;
            let mut actual = blk;
            assert!(hadamard_wht_inplace_i32(&mut expected));
            hadamard_wht_16(&mut actual);
            assert_eq!(actual, expected, "wht16 iter={iter}");
            if (iter & 1) == 1 {
                hadamard_wht_16(&mut actual);
                for (orig, inv) in blk.iter().zip(actual.iter()) {
                    assert_eq!(*inv, orig * 16);
                }
            }
        }
    }

    #[test]
    fn hardware_counter_and_prefetch_smoke() {
        let buf = [42u8; 128];
        prefetch_l1(core::ptr::null());
        prefetch_l1(buf.as_ptr());

        #[cfg(target_arch = "aarch64")]
        {
            let freq = aarch64::read_cntfrq_el0();
            assert!(freq > 0, "cntfrq_el0 must be positive");
            assert_eq!(hardware_counter_freq_hz(), freq);
            let t0 = aarch64::read_cntvct_el0();
            let mut spin = 0u64;
            for i in 0..1_000u64 {
                spin = splitmix64(spin ^ i);
            }
            core::hint::black_box(spin);
            let t1 = aarch64::read_cntvct_el0();
            assert!(t1 >= t0, "cntvct_el0 went backwards: t0={t0}, t1={t1}");
            let h0 = hardware_counter();
            let h1 = hardware_counter();
            assert!(h1 >= h0);
        }
    }

    #[test]
    fn anyrate_2plane_gemv_matches_scalar_oracle_for_1_and_2_planes() {
        let rows = 13usize;
        let cols = 197usize;
        let row_bytes = cols.div_ceil(4);
        let mut p0 = vec![0u8; rows * row_bytes];
        let mut p1 = vec![0u8; rows * row_bytes];
        let mut s0 = vec![0i32; rows];
        let mut s1 = vec![0i32; rows];
        let mut x = vec![0i8; cols];

        let mut seed = 0xCAFE_BABE_1234_5678u64;
        for xi in &mut x {
            seed = splitmix64(seed);
            *xi = (seed & 0xFF) as i8;
        }
        for r in 0..rows {
            seed = splitmix64(seed);
            s0[r] = ((seed & 0x1FFF) as i32) + 4096;
            seed = splitmix64(seed);
            s1[r] = ((seed & 0x0FFF) as i32) + 1024;
            for b in 0..row_bytes {
                seed = splitmix64(seed);
                p0[r * row_bytes + b] = (seed & 0xFF) as u8;
                seed = splitmix64(seed);
                p1[r * row_bytes + b] = (seed & 0xFF) as u8;
            }
        }

        let mut y0_ref = vec![0i32; rows];
        let mut y1_ref = vec![0i32; rows];
        mlx_quant_core::ternary::ternary_2bit_gemv_scalar(&p0, &x, &mut y0_ref, rows, cols);
        mlx_quant_core::ternary::ternary_2bit_gemv_scalar(&p1, &x, &mut y1_ref, rows, cols);

        for active_planes in [1u8, 2u8] {
            let mut actual = vec![0i64; rows];
            anyrate_2plane_gemv(&p0, &p1, &s0, &s1, &x, &mut actual, rows, cols, active_planes);
            for r in 0..rows {
                let mut exp = (y0_ref[r] as i128) * (s0[r] as i128);
                if active_planes >= 2 {
                    exp += (y1_ref[r] as i128) * (s1[r] as i128);
                }
                assert_eq!(actual[r], (exp >> 16) as i64, "row={r} planes={active_planes}");
            }
        }
    }
}
