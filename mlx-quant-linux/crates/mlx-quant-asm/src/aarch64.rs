//! AArch64 NEON & `FEAT_DotProd` inline-assembly compute kernels and hardware counters.
//!
//! All kernels strictly preserve AAPCS64 callee-saved SIMD registers `v8..v15`,
//! execute zero floating-point instructions, and maintain bit-for-bit integer parity
//! with `mlx_quant_core` scalar oracles across all `i8::MIN`/`i8::MAX` edge cases.

/// Baseline AArch64 NEON 64-lane balanced-ternary dot product (`smull`/`smlal` + `saddlv`).
///
/// Compiles without `FEAT_DotProd` (e.g. `aarch64-unknown-none`). Preserves `v8..v15`.
#[cfg(any(not(target_feature = "dotprod"), test))]
#[inline]
pub fn ternary_dot_64_neon_baseline(weights: &[i8; 64], activations: &[i8; 64]) -> i32 {
    let out: i32;
    // SAFETY:
    // - `weights` and `activations` are `&[i8; 64]` (64 valid readable bytes each).
    // - Touches only caller-saved SIMD registers `v0..v7` and `v16..v21` (preserves `v8..v15`).
    // - Intermediate widening products fit cleanly in `i16` (`|lane| <= 512`), then `saddlv` -> `i32`.
    unsafe {
        core::arch::asm!(
            "ld1 {{v0.16b, v1.16b, v2.16b, v3.16b}}, [{w}]",
            "ld1 {{v4.16b, v5.16b, v6.16b, v7.16b}}, [{a}]",
            "movi v20.16b, #1",
            "movi v21.16b, #0xff",
            "smin v0.16b, v0.16b, v20.16b",
            "smin v1.16b, v1.16b, v20.16b",
            "smax v0.16b, v0.16b, v21.16b",
            "smax v1.16b, v1.16b, v21.16b",
            "smin v2.16b, v2.16b, v20.16b",
            "smin v3.16b, v3.16b, v20.16b",
            "smull v16.8h, v0.8b, v4.8b",
            "smull2 v17.8h, v0.16b, v4.16b",
            "smull v18.8h, v1.8b, v5.8b",
            "smull2 v19.8h, v1.16b, v5.16b",
            "smax v2.16b, v2.16b, v21.16b",
            "smax v3.16b, v3.16b, v21.16b",
            "smlal v16.8h, v2.8b, v6.8b",
            "smlal2 v17.8h, v2.16b, v6.16b",
            "smlal v18.8h, v3.8b, v7.8b",
            "smlal2 v19.8h, v3.16b, v7.16b",
            "add v16.8h, v16.8h, v18.8h",
            "add v17.8h, v17.8h, v19.8h",
            "add v16.8h, v16.8h, v17.8h",
            "saddlv s16, v16.8h",
            "umov {out:w}, v16.s[0]",
            w = in(reg) weights.as_ptr(),
            a = in(reg) activations.as_ptr(),
            out = lateout(reg) out,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _,
            options(nostack, readonly, pure, preserves_flags)
        );
    }
    out
}

/// AArch64 `FEAT_DotProd` 64-lane balanced-ternary dot product (`sdot` dual accumulator `v16`/`v17`).
///
/// Preserves `v8..v15`.
#[cfg(target_feature = "dotprod")]
#[inline]
pub fn ternary_dot_64_neon_dotprod(weights: &[i8; 64], activations: &[i8; 64]) -> i32 {
    let out: i32;
    // SAFETY:
    // - `weights` and `activations` are `&[i8; 64]` (64 valid readable bytes each).
    // - Gated on `target_feature = "dotprod"`.
    // - Touches only caller-saved SIMD registers `v0..v7`, `v16..v17`, `v20..v21` (never `v8..v15`).
    unsafe {
        core::arch::asm!(
            "movi v16.4s, #0",
            "movi v17.4s, #0",
            "movi v20.16b, #1",
            "movi v21.16b, #0xff",
            "ld1 {{v0.16b, v1.16b, v2.16b, v3.16b}}, [{w}]",
            "ld1 {{v4.16b, v5.16b, v6.16b, v7.16b}}, [{a}]",
            "smin v0.16b, v0.16b, v20.16b",
            "smin v1.16b, v1.16b, v20.16b",
            "smin v2.16b, v2.16b, v20.16b",
            "smin v3.16b, v3.16b, v20.16b",
            "smax v0.16b, v0.16b, v21.16b",
            "smax v1.16b, v1.16b, v21.16b",
            "smax v2.16b, v2.16b, v21.16b",
            "smax v3.16b, v3.16b, v21.16b",
            "sdot v16.4s, v0.16b, v4.16b",
            "sdot v17.4s, v1.16b, v5.16b",
            "sdot v16.4s, v2.16b, v6.16b",
            "sdot v17.4s, v3.16b, v7.16b",
            "add v16.4s, v16.4s, v17.4s",
            "addv s16, v16.4s",
            "umov {out:w}, v16.s[0]",
            w = in(reg) weights.as_ptr(),
            a = in(reg) activations.as_ptr(),
            out = lateout(reg) out,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v16") _, out("v17") _, out("v20") _, out("v21") _,
            options(nostack, readonly, pure, preserves_flags)
        );
    }
    out
}

/// Baseline NEON 2-bit packed ternary dot product over 64 weights (`{0,1,2,3} -> {-1,0,+1,0}`).
///
/// Unpacks 16 packed bytes inside NEON registers alongside `ld4` deinterleaved activations
/// and widens with `smull`/`smlal` + `saddlv`. Preserves `v8..v15`.
#[cfg(any(not(target_feature = "dotprod"), test))]
#[inline]
pub fn ternary_2bit_dot_64_neon_baseline(packed: &[u8; 16], activations: &[i8; 64]) -> i32 {
    let out: i32;
    // SAFETY:
    // - `packed` is 16 readable bytes and `activations` is 64 readable bytes.
    // - Touches only caller-saved SIMD registers `v0..v7` and `v16..v25` (never `v8..v15`).
    unsafe {
        core::arch::asm!(
            "ld1 {{v0.16b}}, [{p}]",
            "ld4 {{v4.16b, v5.16b, v6.16b, v7.16b}}, [{a}]",
            "movi v20.16b, #3",
            "movi v21.16b, #1",
            "ushr v1.16b, v0.16b, #2",
            "ushr v2.16b, v0.16b, #4",
            "ushr v3.16b, v0.16b, #6",
            "and v0.16b, v0.16b, v20.16b",
            "and v1.16b, v1.16b, v20.16b",
            "and v2.16b, v2.16b, v20.16b",
            "cmeq v22.16b, v0.16b, v20.16b",
            "cmeq v23.16b, v1.16b, v20.16b",
            "cmeq v24.16b, v2.16b, v20.16b",
            "cmeq v25.16b, v3.16b, v20.16b",
            "sub v0.16b, v0.16b, v21.16b",
            "sub v1.16b, v1.16b, v21.16b",
            "sub v2.16b, v2.16b, v21.16b",
            "sub v3.16b, v3.16b, v21.16b",
            "bic v0.16b, v0.16b, v22.16b",
            "bic v1.16b, v1.16b, v23.16b",
            "bic v2.16b, v2.16b, v24.16b",
            "bic v3.16b, v3.16b, v25.16b",
            "smull v16.8h, v0.8b, v4.8b",
            "smull2 v17.8h, v0.16b, v4.16b",
            "smull v18.8h, v1.8b, v5.8b",
            "smull2 v19.8h, v1.16b, v5.16b",
            "smlal v16.8h, v2.8b, v6.8b",
            "smlal2 v17.8h, v2.16b, v6.16b",
            "smlal v18.8h, v3.8b, v7.8b",
            "smlal2 v19.8h, v3.16b, v7.16b",
            "add v16.8h, v16.8h, v18.8h",
            "add v17.8h, v17.8h, v19.8h",
            "add v16.8h, v16.8h, v17.8h",
            "saddlv s16, v16.8h",
            "umov {out:w}, v16.s[0]",
            p = in(reg) packed.as_ptr(),
            a = in(reg) activations.as_ptr(),
            out = lateout(reg) out,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _,
            options(nostack, readonly, pure, preserves_flags)
        );
    }
    out
}

/// `FEAT_DotProd` NEON 2-bit packed ternary dot product over 64 weights (`{0,1,2,3} -> {-1,0,+1,0}`).
#[cfg(target_feature = "dotprod")]
#[inline]
pub fn ternary_2bit_dot_64_neon_dotprod(packed: &[u8; 16], activations: &[i8; 64]) -> i32 {
    let out: i32;
    // SAFETY:
    // - `packed` is 16 readable bytes and `activations` is 64 readable bytes.
    // - Gated on `target_feature = "dotprod"`.
    // - Touches only caller-saved SIMD registers `v0..v7`, `v16..v17`, `v20..v25` (never `v8..v15`).
    unsafe {
        core::arch::asm!(
            "ld1 {{v0.16b}}, [{p}]",
            "ld4 {{v4.16b, v5.16b, v6.16b, v7.16b}}, [{a}]",
            "movi v20.16b, #3",
            "movi v21.16b, #1",
            "ushr v1.16b, v0.16b, #2",
            "ushr v2.16b, v0.16b, #4",
            "ushr v3.16b, v0.16b, #6",
            "and v0.16b, v0.16b, v20.16b",
            "and v1.16b, v1.16b, v20.16b",
            "and v2.16b, v2.16b, v20.16b",
            "cmeq v22.16b, v0.16b, v20.16b",
            "cmeq v23.16b, v1.16b, v20.16b",
            "cmeq v24.16b, v2.16b, v20.16b",
            "cmeq v25.16b, v3.16b, v20.16b",
            "sub v0.16b, v0.16b, v21.16b",
            "sub v1.16b, v1.16b, v21.16b",
            "sub v2.16b, v2.16b, v21.16b",
            "sub v3.16b, v3.16b, v21.16b",
            "bic v0.16b, v0.16b, v22.16b",
            "bic v1.16b, v1.16b, v23.16b",
            "bic v2.16b, v2.16b, v24.16b",
            "bic v3.16b, v3.16b, v25.16b",
            "movi v16.4s, #0",
            "movi v17.4s, #0",
            "sdot v16.4s, v0.16b, v4.16b",
            "sdot v17.4s, v1.16b, v5.16b",
            "sdot v16.4s, v2.16b, v6.16b",
            "sdot v17.4s, v3.16b, v7.16b",
            "add v16.4s, v16.4s, v17.4s",
            "addv s16, v16.4s",
            "umov {out:w}, v16.s[0]",
            p = in(reg) packed.as_ptr(),
            a = in(reg) activations.as_ptr(),
            out = lateout(reg) out,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v16") _, out("v17") _, out("v20") _, out("v21") _,
            out("v22") _, out("v23") _, out("v24") _, out("v25") _,
            options(nostack, readonly, pure, preserves_flags)
        );
    }
    out
}

/// AArch64 NEON 2-bit packed ternary dot product over 64 LSB-first codes `{0,1,2} -> {-1,0,+1}` (`3 -> 0`).
#[inline]
pub fn ternary_2bit_dot_64_neon(packed: &[u8; 16], activations: &[i8; 64]) -> i32 {
    #[cfg(target_feature = "dotprod")]
    {
        ternary_2bit_dot_64_neon_dotprod(packed, activations)
    }
    #[cfg(not(target_feature = "dotprod"))]
    {
        ternary_2bit_dot_64_neon_baseline(packed, activations)
    }
}

/// AArch64 NEON 2-bit packed ternary GEMV (`y_out[..rows] = W * x`).
///
/// Uses [`ternary_2bit_dot_64_neon`] for 64-column blocks with `prfm pldl1keep` cache prefetching
/// and exact scalar tail handling for `cols % 64 != 0`.
pub fn ternary_2bit_gemv_neon(
    w_packed: &[u8],
    x: &[i8],
    y_out: &mut [i32],
    rows: usize,
    cols: usize,
) {
    assert!(y_out.len() >= rows, "y_out buffer too small");
    assert!(x.len() >= cols, "x activation slice too small");
    let row_bytes = cols.div_ceil(4);
    assert!(
        w_packed.len() >= rows.saturating_mul(row_bytes),
        "w_packed buffer too small"
    );

    if rows == 0 {
        return;
    }
    if cols == 0 {
        y_out[..rows].fill(0);
        return;
    }

    let full_blocks = cols / 64;
    let tail_cols = cols - full_blocks * 64;

    let mut r = 0usize;
    while r < rows {
        let row_start = r * row_bytes;
        let row_w = &w_packed[row_start..row_start + row_bytes];
        if r + 1 < rows {
            prefetch_l1_aarch64(w_packed[(r + 1) * row_bytes..].as_ptr());
        }

        let mut acc = 0i32;
        let mut blk = 0usize;
        while blk < full_blocks {
            let byte_off = blk * 16;
            let col_off = blk * 64;
            if byte_off + 64 < row_w.len() {
                prefetch_l1_aarch64(row_w[byte_off + 64..].as_ptr());
            }
            if col_off + 256 < x.len() {
                prefetch_l1_aarch64(x[col_off + 256..].as_ptr().cast::<u8>());
            }
            let w_blk: &[u8; 16] = (&row_w[byte_off..byte_off + 16])
                .try_into()
                .expect("16-byte packed weight block");
            let x_blk: &[i8; 64] = (&x[col_off..col_off + 64])
                .try_into()
                .expect("64-element activation block");
            acc += ternary_2bit_dot_64_neon(w_blk, x_blk);
            blk += 1;
        }

        if tail_cols > 0 {
            let base_byte = full_blocks * 16;
            let base_col = full_blocks * 64;
            let mut c = 0usize;
            while c < tail_cols {
                let byte = row_w[base_byte + (c >> 2)];
                let code = (byte >> ((c & 3) << 1)) & 0x03;
                let w_val = match code {
                    0 => -1i32,
                    2 => 1i32,
                    _ => 0i32,
                };
                acc += w_val * (x[base_col + c] as i32);
                c += 1;
            }
        }

        y_out[r] = acc;
        r += 1;
    }
}

/// AArch64 NEON **Any-Rate 2-Plane Progressive Residual Ternary** dot product (`arXiv:2611.34402`)
/// over 64-lane 2-bit packed planes in zero-float Q15.16 fixed-point.
///
/// - `active_planes == 1`: executes ONLY Plane 0 (`1.6 bpw` Base-243 / 2-bit) via bare-metal NEON
///   (`sdot` on `FEAT_DotProd` or `smull`/`sadalp` baseline).
/// - `active_planes >= 2`: fuses Plane 0 + Plane 1 (`3.2 bpw` cumulative) with Q15.16 scale shift.
#[inline]
pub fn anyrate_2plane_dot_64_neon(
    plane0: &[u8; 16],
    plane1: &[u8; 16],
    activations: &[i8; 64],
    scale0_q16: i32,
    scale1_q16: i32,
    active_planes: u8,
) -> i64 {
    let acc0 = ternary_2bit_dot_64_neon(plane0, activations) as i128;
    let mut total = acc0 * (scale0_q16 as i128);
    if active_planes >= 2 {
        let acc1 = ternary_2bit_dot_64_neon(plane1, activations) as i128;
        total += acc1 * (scale1_q16 as i128);
    }
    (total >> 16) as i64
}

/// AArch64 NEON **Any-Rate 2-Plane Progressive Ternary GEMV** (`y_out_q16[r] = (s0*W0 + s1*W1) x >> 16`).
///
/// Streams L1-prefetched 64-column NEON blocks (`prfm pldl1keep`) from ONE nested 2-plane buffer;
/// callers switch between `active_planes = 1` (fast coarse) and `active_planes = 2` (high accuracy)
/// at runtime with zero re-quantization and zero floats (`#![no_std]`).
#[allow(clippy::too_many_arguments)]
pub fn anyrate_2plane_gemv_neon(
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
    assert!(y_out_q16.len() >= rows, "y_out_q16 buffer too small");
    assert!(scales0_q16.len() >= rows, "scales0_q16 buffer too small");
    assert!(scales1_q16.len() >= rows, "scales1_q16 buffer too small");
    assert!(x.len() >= cols, "x activation slice too small");
    let row_bytes = cols.div_ceil(4);
    assert!(w_plane0.len() >= rows.saturating_mul(row_bytes));
    assert!(w_plane1.len() >= rows.saturating_mul(row_bytes));

    if rows == 0 {
        return;
    }
    if cols == 0 {
        y_out_q16[..rows].fill(0);
        return;
    }

    let full_blocks = cols / 64;
    let tail_cols = cols - full_blocks * 64;
    let use_p1 = active_planes >= 2;

    let mut r = 0usize;
    while r < rows {
        let row_start = r * row_bytes;
        let row0 = &w_plane0[row_start..row_start + row_bytes];
        let row1 = &w_plane1[row_start..row_start + row_bytes];
        if r + 1 < rows {
            prefetch_l1_aarch64(w_plane0[(r + 1) * row_bytes..].as_ptr());
            if use_p1 {
                prefetch_l1_aarch64(w_plane1[(r + 1) * row_bytes..].as_ptr());
            }
        }

        let mut acc0 = 0i64;
        let mut acc1 = 0i64;
        let mut blk = 0usize;
        while blk < full_blocks {
            let bo = blk * 16;
            let co = blk * 64;
            let x_blk: &[i8; 64] = (&x[co..co + 64]).try_into().expect("64-lane x");
            let w0_blk: &[u8; 16] = (&row0[bo..bo + 16]).try_into().expect("16-byte w0");
            acc0 += ternary_2bit_dot_64_neon(w0_blk, x_blk) as i64;
            if use_p1 {
                let w1_blk: &[u8; 16] = (&row1[bo..bo + 16]).try_into().expect("16-byte w1");
                acc1 += ternary_2bit_dot_64_neon(w1_blk, x_blk) as i64;
            }
            blk += 1;
        }

        if tail_cols > 0 {
            let base_byte = full_blocks * 16;
            let base_col = full_blocks * 64;
            let mut c = 0usize;
            while c < tail_cols {
                let shift = (c & 3) << 1;
                let xi = x[base_col + c] as i64;
                let c0 = (row0[base_byte + (c >> 2)] >> shift) & 0x03;
                acc0 += match c0 {
                    0 => -xi,
                    2 => xi,
                    _ => 0,
                };
                if use_p1 {
                    let c1 = (row1[base_byte + (c >> 2)] >> shift) & 0x03;
                    acc1 += match c1 {
                        0 => -xi,
                        2 => xi,
                        _ => 0,
                    };
                }
                c += 1;
            }
        }

        let mut total = (acc0 as i128) * (scales0_q16[r] as i128);
        if use_p1 {
            total += (acc1 as i128) * (scales1_q16[r] as i128);
        }
        y_out_q16[r] = (total >> 16) as i64;
        r += 1;
    }
}

/// Baseline NEON 32-element INT4 signed-nibble dot product (16 packed bytes per operand).
///
/// Sign-extends low nibbles `[3:0]` via `shl #4` / `sshr #4` and high nibbles `[7:4]` via
/// `sshr #4` inside NEON registers, then accumulates via `smull`/`smlal` + `saddlv`.
#[cfg(any(not(target_feature = "dotprod"), test))]
#[inline]
pub fn int4_dot_32_neon_baseline(a: &[u8; 16], b: &[u8; 16]) -> i32 {
    let out: i32;
    // SAFETY:
    // - `a` and `b` are 16 valid readable bytes each.
    // - Touches only caller-saved SIMD registers `v0..v5` and `v16..v19` (never `v8..v15`).
    unsafe {
        core::arch::asm!(
            "ld1 {{v0.16b}}, [{a}]",
            "ld1 {{v1.16b}}, [{b}]",
            "shl v2.16b, v0.16b, #4",
            "sshr v2.16b, v2.16b, #4",
            "sshr v3.16b, v0.16b, #4",
            "shl v4.16b, v1.16b, #4",
            "sshr v4.16b, v4.16b, #4",
            "sshr v5.16b, v1.16b, #4",
            "smull v16.8h, v2.8b, v4.8b",
            "smull2 v17.8h, v2.16b, v4.16b",
            "smull v18.8h, v3.8b, v5.8b",
            "smull2 v19.8h, v3.16b, v5.16b",
            "add v16.8h, v16.8h, v18.8h",
            "add v17.8h, v17.8h, v19.8h",
            "add v16.8h, v16.8h, v17.8h",
            "saddlv s16, v16.8h",
            "umov {out:w}, v16.s[0]",
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            out = lateout(reg) out,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            options(nostack, readonly, pure, preserves_flags)
        );
    }
    out
}

/// `FEAT_DotProd` NEON 32-element INT4 signed-nibble dot product (16 packed bytes per operand).
#[cfg(target_feature = "dotprod")]
#[inline]
pub fn int4_dot_32_neon_dotprod(a: &[u8; 16], b: &[u8; 16]) -> i32 {
    let out: i32;
    // SAFETY:
    // - `a` and `b` are 16 valid readable bytes each.
    // - Gated on `target_feature = "dotprod"`.
    // - Touches only caller-saved SIMD registers `v0..v5` and `v16..v17` (never `v8..v15`).
    unsafe {
        core::arch::asm!(
            "ld1 {{v0.16b}}, [{a}]",
            "ld1 {{v1.16b}}, [{b}]",
            "shl v2.16b, v0.16b, #4",
            "sshr v2.16b, v2.16b, #4",
            "sshr v3.16b, v0.16b, #4",
            "shl v4.16b, v1.16b, #4",
            "sshr v4.16b, v4.16b, #4",
            "sshr v5.16b, v1.16b, #4",
            "movi v16.4s, #0",
            "movi v17.4s, #0",
            "sdot v16.4s, v2.16b, v4.16b",
            "sdot v17.4s, v3.16b, v5.16b",
            "add v16.4s, v16.4s, v17.4s",
            "addv s16, v16.4s",
            "umov {out:w}, v16.s[0]",
            a = in(reg) a.as_ptr(),
            b = in(reg) b.as_ptr(),
            out = lateout(reg) out,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _,
            out("v16") _, out("v17") _,
            options(nostack, readonly, pure, preserves_flags)
        );
    }
    out
}

#[inline]
fn int4_dot_32_neon(a: &[u8; 16], b: &[u8; 16]) -> i32 {
    #[cfg(target_feature = "dotprod")]
    {
        int4_dot_32_neon_dotprod(a, b)
    }
    #[cfg(not(target_feature = "dotprod"))]
    {
        int4_dot_32_neon_baseline(a, b)
    }
}

/// 2x4 tiled `sdot` microkernel over `chunks > 0` 16-byte INT4 blocks.
///
/// # Safety
/// - `chunks > 0`.
/// - `a_ptrs[0..2]` and `b_ptrs[0..4]` each point to at least `chunks * 16` readable bytes.
/// - `out[0..2]` each point to 4 writable `i32` elements.
#[cfg(target_feature = "dotprod")]
#[inline]
unsafe fn int4_tile_2x4_chunks_dotprod(
    a_ptrs: [*const u8; 2],
    b_ptrs: [*const u8; 4],
    mut chunks: usize,
    out: [*mut i32; 2],
) {
    let [mut a0, mut a1] = a_ptrs;
    let [mut b0, mut b1, mut b2, mut b3] = b_ptrs;
    // SAFETY: Caller guarantees valid pointers for `chunks * 16` bytes and 4-element outputs.
    // Preserves `v8..v15` (uses `v0..v7` and `v16..v29`).
    unsafe {
        core::arch::asm!(
            "movi v20.4s, #0",
            "movi v21.4s, #0",
            "movi v22.4s, #0",
            "movi v23.4s, #0",
            "movi v24.4s, #0",
            "movi v25.4s, #0",
            "movi v26.4s, #0",
            "movi v27.4s, #0",
            "2:",
            "ld1 {{v0.16b}}, [{a0}], #16",
            "ld1 {{v1.16b}}, [{a1}], #16",
            "ld1 {{v4.16b}}, [{b0}], #16",
            "ld1 {{v5.16b}}, [{b1}], #16",
            "ld1 {{v6.16b}}, [{b2}], #16",
            "ld1 {{v7.16b}}, [{b3}], #16",
            "sshr v28.16b, v0.16b, #4",
            "shl v0.16b, v0.16b, #4",
            "sshr v0.16b, v0.16b, #4",
            "sshr v29.16b, v1.16b, #4",
            "shl v1.16b, v1.16b, #4",
            "sshr v1.16b, v1.16b, #4",
            "sshr v16.16b, v4.16b, #4",
            "shl v4.16b, v4.16b, #4",
            "sshr v4.16b, v4.16b, #4",
            "sshr v17.16b, v5.16b, #4",
            "shl v5.16b, v5.16b, #4",
            "sshr v5.16b, v5.16b, #4",
            "sshr v18.16b, v6.16b, #4",
            "shl v6.16b, v6.16b, #4",
            "sshr v6.16b, v6.16b, #4",
            "sshr v19.16b, v7.16b, #4",
            "shl v7.16b, v7.16b, #4",
            "sshr v7.16b, v7.16b, #4",
            "sdot v20.4s, v0.16b, v4.16b",
            "sdot v20.4s, v28.16b, v16.16b",
            "sdot v21.4s, v0.16b, v5.16b",
            "sdot v21.4s, v28.16b, v17.16b",
            "sdot v22.4s, v0.16b, v6.16b",
            "sdot v22.4s, v28.16b, v18.16b",
            "sdot v23.4s, v0.16b, v7.16b",
            "sdot v23.4s, v28.16b, v19.16b",
            "sdot v24.4s, v1.16b, v4.16b",
            "sdot v24.4s, v29.16b, v16.16b",
            "sdot v25.4s, v1.16b, v5.16b",
            "sdot v25.4s, v29.16b, v17.16b",
            "sdot v26.4s, v1.16b, v6.16b",
            "sdot v26.4s, v29.16b, v18.16b",
            "sdot v27.4s, v1.16b, v7.16b",
            "sdot v27.4s, v29.16b, v19.16b",
            "subs {cnt}, {cnt}, #1",
            "b.ne 2b",
            "addp v20.4s, v20.4s, v21.4s",
            "addp v22.4s, v22.4s, v23.4s",
            "addp v20.4s, v20.4s, v22.4s",
            "addp v24.4s, v24.4s, v25.4s",
            "addp v26.4s, v26.4s, v27.4s",
            "addp v24.4s, v24.4s, v26.4s",
            "st1 {{v20.4s}}, [{out0}]",
            "st1 {{v24.4s}}, [{out1}]",
            a0 = inout(reg) a0,
            a1 = inout(reg) a1,
            b0 = inout(reg) b0,
            b1 = inout(reg) b1,
            b2 = inout(reg) b2,
            b3 = inout(reg) b3,
            cnt = inout(reg) chunks,
            out0 = in(reg) out[0],
            out1 = in(reg) out[1],
            out("v0") _, out("v1") _, out("v4") _, out("v5") _,
            out("v6") _, out("v7") _, out("v16") _, out("v17") _,
            out("v18") _, out("v19") _, out("v20") _, out("v21") _,
            out("v22") _, out("v23") _, out("v24") _, out("v25") _,
            out("v26") _, out("v27") _, out("v28") _, out("v29") _,
            options(nostack)
        );
    }
    let _ = (a0, a1, b0, b1, b2, b3, chunks);
}

#[inline]
fn int4_tail_dot_neon(a_tail: &[u8], b_tail: &[u8], k_is_odd: bool) -> i32 {
    if a_tail.is_empty() {
        return 0;
    }
    let mut a_pad = [0u8; 16];
    let mut b_pad = [0u8; 16];
    let len = a_tail.len();
    a_pad[..len].copy_from_slice(a_tail);
    b_pad[..len].copy_from_slice(b_tail);
    if k_is_odd {
        // Zero unused high nibble [7:4] of the final packed byte.
        a_pad[len - 1] &= 0x0F;
        b_pad[len - 1] &= 0x0F;
    }
    int4_dot_32_neon(&a_pad, &b_pad)
}

#[inline]
fn int4_dot_row_neon(a_row: &[u8], b_row: &[u8], full_chunks: usize, k_is_odd: bool) -> i32 {
    let mut acc = 0i32;
    let mut chunk = 0usize;
    while chunk < full_chunks {
        let off = chunk * 16;
        let a_blk: &[u8; 16] = (&a_row[off..off + 16])
            .try_into()
            .expect("16-byte INT4 A chunk");
        let b_blk: &[u8; 16] = (&b_row[off..off + 16])
            .try_into()
            .expect("16-byte INT4 B chunk");
        acc += int4_dot_32_neon(a_blk, b_blk);
        chunk += 1;
    }
    let tail_off = full_chunks * 16;
    if tail_off < a_row.len() {
        acc += int4_tail_dot_neon(&a_row[tail_off..], &b_row[tail_off..], k_is_odd);
    }
    acc
}

/// Complete tiled AArch64 NEON INT4 signed-nibble GEMM (`C = A * B^T`, low nibble `[3:0]` first).
///
/// - `a_packed`: `m x ceil_div(k, 2)` row-major bytes
/// - `b_t_packed`: `n x ceil_div(k, 2)` transposed row-major bytes
/// - `c_out`: `m x n` row-major `i32` output matching `int4_gemm_scalar` bit-for-bit.
pub fn int4_gemm_neon(
    a_packed: &[u8],
    b_t_packed: &[u8],
    c_out: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert!(c_out.len() >= m.saturating_mul(n), "c_out buffer too small");
    let packed_k = k.div_ceil(2);
    assert!(
        a_packed.len() >= m.saturating_mul(packed_k),
        "a_packed buffer too small"
    );
    assert!(
        b_t_packed.len() >= n.saturating_mul(packed_k),
        "b_t_packed buffer too small"
    );

    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        c_out[..m * n].fill(0);
        return;
    }

    let full_chunks = packed_k / 16;
    let tail_off = full_chunks * 16;
    let k_is_odd = (k & 1) != 0;

    let mut r = 0usize;
    while r + 2 <= m {
        let a0_row = &a_packed[r * packed_k..(r + 1) * packed_k];
        let a1_row = &a_packed[(r + 1) * packed_k..(r + 2) * packed_k];
        let mut col = 0usize;

        while col + 4 <= n {
            let b_rows = [
                &b_t_packed[col * packed_k..(col + 1) * packed_k],
                &b_t_packed[(col + 1) * packed_k..(col + 2) * packed_k],
                &b_t_packed[(col + 2) * packed_k..(col + 3) * packed_k],
                &b_t_packed[(col + 3) * packed_k..(col + 4) * packed_k],
            ];

            let mut tile_r0 = [0i32; 4];
            let mut tile_r1 = [0i32; 4];

            if full_chunks > 0 {
                #[cfg(target_feature = "dotprod")]
                // SAFETY: Each slice has at least `packed_k >= full_chunks * 16` bytes,
                // and `tile_r0` / `tile_r1` hold 4 `i32` elements.
                unsafe {
                    int4_tile_2x4_chunks_dotprod(
                        [a0_row.as_ptr(), a1_row.as_ptr()],
                        [
                            b_rows[0].as_ptr(),
                            b_rows[1].as_ptr(),
                            b_rows[2].as_ptr(),
                            b_rows[3].as_ptr(),
                        ],
                        full_chunks,
                        [tile_r0.as_mut_ptr(), tile_r1.as_mut_ptr()],
                    );
                }
                #[cfg(not(target_feature = "dotprod"))]
                {
                    let mut chunk = 0usize;
                    while chunk < full_chunks {
                        let off = chunk * 16;
                        let a0_c: &[u8; 16] = (&a0_row[off..off + 16]).try_into().unwrap();
                        let a1_c: &[u8; 16] = (&a1_row[off..off + 16]).try_into().unwrap();
                        let mut j = 0usize;
                        while j < 4 {
                            let bj_c: &[u8; 16] = (&b_rows[j][off..off + 16]).try_into().unwrap();
                            tile_r0[j] += int4_dot_32_neon(a0_c, bj_c);
                            tile_r1[j] += int4_dot_32_neon(a1_c, bj_c);
                            j += 1;
                        }
                        chunk += 1;
                    }
                }
            }

            if tail_off < packed_k {
                let mut j = 0usize;
                while j < 4 {
                    tile_r0[j] +=
                        int4_tail_dot_neon(&a0_row[tail_off..], &b_rows[j][tail_off..], k_is_odd);
                    tile_r1[j] +=
                        int4_tail_dot_neon(&a1_row[tail_off..], &b_rows[j][tail_off..], k_is_odd);
                    j += 1;
                }
            }

            let out_r0 = r * n + col;
            let out_r1 = (r + 1) * n + col;
            c_out[out_r0..out_r0 + 4].copy_from_slice(&tile_r0);
            c_out[out_r1..out_r1 + 4].copy_from_slice(&tile_r1);
            col += 4;
        }

        while col < n {
            let b_row = &b_t_packed[col * packed_k..(col + 1) * packed_k];
            c_out[r * n + col] = int4_dot_row_neon(a0_row, b_row, full_chunks, k_is_odd);
            c_out[(r + 1) * n + col] = int4_dot_row_neon(a1_row, b_row, full_chunks, k_is_odd);
            col += 1;
        }

        r += 2;
    }

    while r < m {
        let a_row = &a_packed[r * packed_k..(r + 1) * packed_k];
        let mut col = 0usize;
        while col < n {
            let b_row = &b_t_packed[col * packed_k..(col + 1) * packed_k];
            c_out[r * n + col] = int4_dot_row_neon(a_row, b_row, full_chunks, k_is_odd);
            col += 1;
        }
        r += 1;
    }
}

/// 4-stage AArch64 NEON 16-element Fast Walsh-Hadamard Transform (`sqadd`/`sqsub` butterfly).
///
/// Matches `mlx_quant_core::ternary::hadamard_wht_inplace_i32` bit-for-bit across all inputs
/// including `i32::MIN` and `i32::MAX` saturation boundaries. Preserves `v8..v15`.
#[inline]
pub fn hadamard_wht_16_neon(block: &mut [i32; 16]) {
    // SAFETY:
    // - `block` is an exclusive `&mut [i32; 16]` (64 valid readable/writable bytes).
    // - Touches only caller-saved SIMD registers `v0..v7` and `v16..v19` (never `v8..v15`).
    // - `sqadd`/`sqsub` match `i32::saturating_add`/`i32::saturating_sub` bit-for-bit.
    unsafe {
        core::arch::asm!(
            "ld1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{p}]",
            // Stage 1 (h = 1)
            "uzp1 v4.4s, v0.4s, v1.4s",
            "uzp2 v5.4s, v0.4s, v1.4s",
            "uzp1 v6.4s, v2.4s, v3.4s",
            "uzp2 v7.4s, v2.4s, v3.4s",
            "sqadd v16.4s, v4.4s, v5.4s",
            "sqsub v17.4s, v4.4s, v5.4s",
            "sqadd v18.4s, v6.4s, v7.4s",
            "sqsub v19.4s, v6.4s, v7.4s",
            // Stage 2 (h = 2)
            "uzp1 v4.4s, v16.4s, v17.4s",
            "uzp2 v5.4s, v16.4s, v17.4s",
            "uzp1 v6.4s, v18.4s, v19.4s",
            "uzp2 v7.4s, v18.4s, v19.4s",
            "sqadd v16.4s, v4.4s, v5.4s",
            "sqsub v17.4s, v4.4s, v5.4s",
            "sqadd v18.4s, v6.4s, v7.4s",
            "sqsub v19.4s, v6.4s, v7.4s",
            // Stage 3 (h = 4)
            "uzp1 v4.4s, v16.4s, v17.4s",
            "uzp2 v5.4s, v16.4s, v17.4s",
            "uzp1 v6.4s, v18.4s, v19.4s",
            "uzp2 v7.4s, v18.4s, v19.4s",
            "sqadd v0.4s, v4.4s, v5.4s",
            "sqsub v1.4s, v4.4s, v5.4s",
            "sqadd v2.4s, v6.4s, v7.4s",
            "sqsub v3.4s, v6.4s, v7.4s",
            // Stage 4 (h = 8)
            "sqadd v4.4s, v0.4s, v2.4s",
            "sqadd v5.4s, v1.4s, v3.4s",
            "sqsub v6.4s, v0.4s, v2.4s",
            "sqsub v7.4s, v1.4s, v3.4s",
            "st1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{p}]",
            p = in(reg) block.as_mut_ptr(),
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            options(nostack)
        );
    }
}

/// Issue a non-faulting AArch64 L1 temporal data cache prefetch (`prfm pldl1keep`).
#[inline]
pub fn prefetch_l1_aarch64(ptr: *const u8) {
    if !ptr.is_null() {
        // SAFETY: `prfm pldl1keep` is a non-faulting architectural cache prefetch hint.
        unsafe {
            core::arch::asm!(
                "prfm pldl1keep, [{p}]",
                p = in(reg) ptr,
                options(nostack, preserves_flags)
            );
        }
    }
}

/// Read the AArch64 EL0 virtual counter register (`cntvct_el0`) after an `isb` barrier.
#[inline]
pub fn read_cntvct_el0() -> u64 {
    let out: u64;
    // SAFETY: `cntvct_el0` is unprivileged EL0 virtual timer counter; `isb` serializes fetch.
    unsafe {
        core::arch::asm!(
            "isb",
            "mrs {out}, cntvct_el0",
            out = out(reg) out,
            options(nomem, nostack, preserves_flags)
        );
    }
    out
}

/// Read the AArch64 EL0 counter-timer frequency register (`cntfrq_el0`) in Hz.
#[inline]
pub fn read_cntfrq_el0() -> u64 {
    let out: u64;
    // SAFETY: `cntfrq_el0` is the read-only EL0 counter-timer frequency register on AArch64.
    unsafe {
        core::arch::asm!(
            "mrs {out}, cntfrq_el0",
            out = out(reg) out,
            options(nomem, nostack, preserves_flags)
        );
    }
    out
}
