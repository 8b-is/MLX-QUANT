//! Balanced ternary `{-1, 0, +1}` encodings (OmniLLL & MLX-QUANT `{0,1,2}`),
//! BITCOS sparse packing/GEMV (`arXiv:2609.16338`), Base-243 SPQT, symmetric INT4 packed GEMM,
//! and H-BitLinear Walsh-Hadamard + BitNet b1.58 AbsMean quantizers.
//!
//! Pure `#![no_std]`, zero floats, zero heap allocations.

use crate::ratio::{Q16_ONE, Ratio};

/// Balanced ternary weight / polarity trit in `{-1, 0, +1}`.
#[repr(i8)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Trit {
    Neg = -1,
    #[default]
    Zero = 0,
    Pos = 1,
}

impl Trit {
    /// Normalize any signed byte to a balanced trit `{-1, 0, +1}`.
    #[inline]
    pub const fn from_i8(v: i8) -> Self {
        if v > 0 {
            Self::Pos
        } else if v < 0 {
            Self::Neg
        } else {
            Self::Zero
        }
    }

    /// Return the signed `i8` value in `{-1, 0, +1}`.
    #[inline]
    pub const fn as_i8(self) -> i8 {
        self as i8
    }

    /// Encode in OmniLLL canonical 2-bit format: `Zero = 0b00`, `Pos = 0b01`, `Neg = 0b10`.
    #[inline]
    pub const fn encode_2bit(self) -> u8 {
        match self {
            Self::Zero => 0b00,
            Self::Pos => 0b01,
            Self::Neg => 0b10,
        }
    }

    /// Decode from OmniLLL canonical 2-bit format (`0b11` maps to `Zero`).
    #[inline]
    pub const fn decode_2bit(bits: u8) -> Self {
        match bits & 0b11 {
            0b01 => Self::Pos,
            0b10 => Self::Neg,
            _ => Self::Zero,
        }
    }

    /// Encode in MLX-QUANT 2-bit format `{0, 1, 2} -> {-1, 0, +1}`:
    /// `Neg = 0b00`, `Zero = 0b01`, `Pos = 0b10`.
    #[inline]
    pub const fn encode_mlx_2bit(self) -> u8 {
        match self {
            Self::Neg => 0b00,
            Self::Zero => 0b01,
            Self::Pos => 0b10,
        }
    }

    /// Decode from MLX-QUANT 2-bit format `{0, 1, 2} -> {-1, 0, +1}` (`0b11` maps to `Zero`).
    #[inline]
    pub const fn decode_mlx_2bit(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => Self::Neg,
            0b10 => Self::Pos,
            _ => Self::Zero,
        }
    }
}

/// Pack 4 trits into 1 byte using OmniLLL 2-bit encoding (`0b00=0, 0b01=+1, 0b10=-1`, LSB first).
#[inline]
pub const fn pack_trits_4(t: [Trit; 4]) -> u8 {
    t[0].encode_2bit()
        | (t[1].encode_2bit() << 2)
        | (t[2].encode_2bit() << 4)
        | (t[3].encode_2bit() << 6)
}

/// Unpack 1 byte from OmniLLL 2-bit encoding into 4 trits.
#[inline]
pub const fn unpack_trits_4(byte: u8) -> [Trit; 4] {
    [
        Trit::decode_2bit(byte),
        Trit::decode_2bit(byte >> 2),
        Trit::decode_2bit(byte >> 4),
        Trit::decode_2bit(byte >> 6),
    ]
}

/// Pack 4 trits into 1 byte using MLX-QUANT 2-bit encoding (`0=-1, 1=0, 2=+1`, LSB first).
#[inline]
pub const fn pack_mlx_ternary_4(t: [Trit; 4]) -> u8 {
    t[0].encode_mlx_2bit()
        | (t[1].encode_mlx_2bit() << 2)
        | (t[2].encode_mlx_2bit() << 4)
        | (t[3].encode_mlx_2bit() << 6)
}

/// Unpack 1 byte from MLX-QUANT 2-bit encoding (`{0, 1, 2} -> {-1, 0, +1}`, `3 -> 0`, LSB first).
#[inline]
pub const fn unpack_mlx_ternary_4(byte: u8) -> [Trit; 4] {
    [
        Trit::decode_mlx_2bit(byte),
        Trit::decode_mlx_2bit(byte >> 2),
        Trit::decode_mlx_2bit(byte >> 4),
        Trit::decode_mlx_2bit(byte >> 6),
    ]
}

/// Pack an arbitrary-length slice of [`Trit`] values in MLX-QUANT 2-bit format (`{0,1,2}`)
/// into `packed_out`, zero-padding any unused lanes in the final byte with `Trit::Zero` (`0b01`).
/// Returns the number of bytes written (`trits.len().div_ceil(4)`).
pub fn pack_mlx_ternary_row(trits: &[Trit], packed_out: &mut [u8]) -> usize {
    let needed = trits.len().div_ceil(4);
    assert!(packed_out.len() >= needed, "packed_out buffer too small");
    let mut b = 0usize;
    while b < needed {
        let base = b * 4;
        let t0 = trits[base];
        let t1 = if base + 1 < trits.len() {
            trits[base + 1]
        } else {
            Trit::Zero
        };
        let t2 = if base + 2 < trits.len() {
            trits[base + 2]
        } else {
            Trit::Zero
        };
        let t3 = if base + 3 < trits.len() {
            trits[base + 3]
        } else {
            Trit::Zero
        };
        packed_out[b] = pack_mlx_ternary_4([t0, t1, t2, t3]);
        b += 1;
    }
    needed
}

/// Unpack `cols` trits from MLX-QUANT 2-bit packed bytes into `trits_out[..cols]`.
pub fn unpack_mlx_ternary_row(packed: &[u8], cols: usize, trits_out: &mut [Trit]) {
    let needed = cols.div_ceil(4);
    assert!(packed.len() >= needed, "packed buffer too small");
    assert!(trits_out.len() >= cols, "trits_out buffer too small");
    let mut c = 0usize;
    while c < cols {
        let byte = packed[c >> 2];
        let code = (byte >> ((c & 3) * 2)) & 0b11;
        trits_out[c] = Trit::decode_mlx_2bit(code);
        c += 1;
    }
}

/// Exact scalar ternary dot product over 64-lane `i8` slices.
///
/// Any `w[i] > 0` acts as `+1`, `w[i] < 0` acts as `-1`, and `0` contributes `0`,
/// matching clamped signum NEON kernels bit-for-bit without `i8::MIN * -1` overflow.
#[inline]
pub const fn ternary_dot_64_scalar(w: &[i8; 64], a: &[i8; 64]) -> i32 {
    let mut acc = 0i32;
    let mut i = 0usize;
    while i < 64 {
        let wi = w[i];
        let ai = a[i] as i32;
        if wi > 0 {
            acc += ai;
        } else if wi < 0 {
            acc -= ai;
        }
        i += 1;
    }
    acc
}

/// Exact arbitrary-length scalar ternary dot product over `i8` slices.
#[inline]
pub fn ternary_dot_scalar(w: &[i8], a: &[i8]) -> i32 {
    assert_eq!(w.len(), a.len(), "weight and activation slices must match");
    let mut acc = 0i32;
    for (&wi, &ai) in w.iter().zip(a.iter()) {
        let act = ai as i32;
        if wi > 0 {
            acc += act;
        } else if wi < 0 {
            acc -= act;
        }
    }
    acc
}

/// Exact scalar 64-lane dot product over MLX-QUANT 2-bit packed weights (`{0,1,2} -> {-1,0,+1}`, `3 -> 0`).
#[inline]
pub const fn ternary_2bit_dot_64_scalar(packed: &[u8; 16], a: &[i8; 64]) -> i32 {
    let mut acc = 0i32;
    let mut b = 0usize;
    while b < 16 {
        let byte = packed[b];
        let base = b << 2;
        let mut lane = 0usize;
        while lane < 4 {
            let code = (byte >> (lane * 2)) & 0b11;
            let act = a[base + lane] as i32;
            match code {
                0b00 => acc -= act,
                0b10 => acc += act,
                _ => {}
            }
            lane += 1;
        }
        b += 1;
    }
    acc
}

/// Exact arbitrary-column scalar dot product over MLX-QUANT 2-bit packed weights (`{0,1,2} -> {-1,0,+1}`).
#[inline]
pub fn ternary_2bit_dot_scalar(packed: &[u8], a: &[i8], cols: usize) -> i32 {
    let needed_bytes = cols.div_ceil(4);
    assert!(packed.len() >= needed_bytes, "packed buffer too small");
    assert!(a.len() >= cols, "activations buffer too small");

    let full_bytes = cols >> 2;
    let mut acc = 0i32;
    let mut b = 0usize;
    while b < full_bytes {
        let byte = packed[b];
        let base = b << 2;
        let c0 = byte & 0b11;
        let c1 = (byte >> 2) & 0b11;
        let c2 = (byte >> 4) & 0b11;
        let c3 = (byte >> 6) & 0b11;

        let s0 = i32::from(c0 == 2) - i32::from(c0 == 0);
        let s1 = i32::from(c1 == 2) - i32::from(c1 == 0);
        let s2 = i32::from(c2 == 2) - i32::from(c2 == 0);
        let s3 = i32::from(c3 == 2) - i32::from(c3 == 0);

        acc += s0 * (a[base] as i32)
            + s1 * (a[base + 1] as i32)
            + s2 * (a[base + 2] as i32)
            + s3 * (a[base + 3] as i32);
        b += 1;
    }

    let rem = cols & 3;
    if rem != 0 {
        let tail_byte = packed[full_bytes];
        let base = full_bytes << 2;
        let mut k = 0usize;
        while k < rem {
            let code = (tail_byte >> (k * 2)) & 0b11;
            let sign = i32::from(code == 2) - i32::from(code == 0);
            acc += sign * (a[base + k] as i32);
            k += 1;
        }
    }
    acc
}

/// Exact scalar GEMV `y_out = W * x` for MLX-QUANT 2-bit packed weights (`row stride = cols.div_ceil(4)`).
pub fn ternary_2bit_gemv_scalar(
    w_packed: &[u8],
    x: &[i8],
    y_out: &mut [i32],
    rows: usize,
    cols: usize,
) {
    let stride = cols.div_ceil(4);
    assert!(
        w_packed.len() >= rows.saturating_mul(stride),
        "w_packed buffer too small"
    );
    assert!(x.len() >= cols, "x buffer too small");
    assert!(y_out.len() >= rows, "y_out buffer too small");

    for (r, dst) in y_out[..rows].iter_mut().enumerate() {
        let row_slice = &w_packed[r * stride..(r + 1) * stride];
        *dst = ternary_2bit_dot_scalar(row_slice, x, cols);
    }
}

/// Pack balanced ternary weights into **BITCOS** (`arXiv:2609.16338`) layout:
/// LSB-first presence bitmap + compacted LSB-first sign stream (`1` = `+1`, `0` = `-1`).
/// Returns `(nnz, sign_bytes_used)`.
pub fn bitcos_pack_into(
    trits: &[Trit],
    presence_out: &mut [u8],
    signs_out: &mut [u8],
) -> (usize, usize) {
    let p_bytes = trits.len().div_ceil(8);
    assert!(
        presence_out.len() >= p_bytes,
        "presence_out buffer too small"
    );

    let nnz = trits
        .iter()
        .filter(|&&t| !matches!(t, Trit::Zero))
        .count();
    let s_bytes = nnz.div_ceil(8);
    assert!(signs_out.len() >= s_bytes, "signs_out buffer too small");

    presence_out[..p_bytes].fill(0);
    signs_out[..s_bytes].fill(0);

    let mut sign_idx = 0usize;
    for (i, &t) in trits.iter().enumerate() {
        match t {
            Trit::Zero => {}
            Trit::Pos => {
                presence_out[i >> 3] |= 1u8 << (i & 7);
                signs_out[sign_idx >> 3] |= 1u8 << (sign_idx & 7);
                sign_idx += 1;
            }
            Trit::Neg => {
                presence_out[i >> 3] |= 1u8 << (i & 7);
                sign_idx += 1;
            }
        }
    }
    (nnz, s_bytes)
}

/// Unpack `len` trits from a BITCOS `(presence, signs)` pair into `trits_out[..len]`.
/// Returns `nnz`.
pub fn bitcos_unpack_into(
    len: usize,
    presence: &[u8],
    signs: &[u8],
    trits_out: &mut [Trit],
) -> usize {
    assert!(trits_out.len() >= len, "trits_out buffer too small");
    let p_bytes = len.div_ceil(8);
    assert!(presence.len() >= p_bytes, "presence buffer too small");

    let mut sign_idx = 0usize;
    for (i, dst) in trits_out[..len].iter_mut().enumerate() {
        let present = (presence[i >> 3] >> (i & 7)) & 1;
        if present == 0 {
            *dst = Trit::Zero;
        } else {
            let s_byte_idx = sign_idx >> 3;
            assert!(s_byte_idx < signs.len(), "signs buffer truncated");
            let sign_bit = (signs[s_byte_idx] >> (sign_idx & 7)) & 1;
            *dst = if sign_bit != 0 { Trit::Pos } else { Trit::Neg };
            sign_idx += 1;
        }
    }
    sign_idx
}

/// Exact integer dot product over a BITCOS-packed row and `i8` activations,
/// skipping zero weights via `presence` bit-scanning (`trailing_zeros`).
pub fn bitcos_dot_i8(len: usize, presence: &[u8], signs: &[u8], activations: &[i8]) -> i32 {
    assert!(activations.len() >= len, "activations slice too small");
    let p_bytes = len.div_ceil(8);
    assert!(presence.len() >= p_bytes, "presence buffer too small");

    let mut acc = 0i32;
    let mut sign_idx = 0usize;
    let rem_bits = len & 7;
    for (b, &raw_p) in presence[..p_bytes].iter().enumerate() {
        let mut p_byte = raw_p;
        if b + 1 == p_bytes && rem_bits != 0 {
            p_byte &= (1u8 << rem_bits) - 1;
        }
        let base = b << 3;
        while p_byte != 0 {
            let bit = p_byte.trailing_zeros() as usize;
            p_byte &= p_byte - 1;
            let col = base + bit;
            let act = activations[col] as i32;
            let s_byte_idx = sign_idx >> 3;
            assert!(s_byte_idx < signs.len(), "signs buffer truncated");
            let sign_bit = (signs[s_byte_idx] >> (sign_idx & 7)) & 1;
            sign_idx += 1;
            if sign_bit != 0 {
                acc += act;
            } else {
                acc -= act;
            }
        }
    }
    acc
}

/// Exact BITCOS effective bits-per-weight: `(len + nnz) / len = 2 - zero_density`.
#[inline]
pub const fn bitcos_effective_bpw(len: usize, nnz: usize) -> Ratio {
    if len == 0 {
        return Ratio::ZERO;
    }
    let total_bits = (len as u64).saturating_add(nnz as u64);
    Ratio::new(total_bits, len as u64)
}

/// Exact scalar BITCOS GEMV `y_out = W_bitcos * x`.
///
/// `presence` has `cols.div_ceil(8)` bytes/row and each row `r` reads its sign bits
/// starting at `signs[row_sign_byte_offsets[r] as usize..]`.
pub fn bitcos_gemv_scalar(
    presence: &[u8],
    signs: &[u8],
    row_sign_byte_offsets: &[u32],
    x: &[i8],
    y_out: &mut [i32],
    rows: usize,
    cols: usize,
) {
    let p_stride = cols.div_ceil(8);
    assert!(
        presence.len() >= rows.saturating_mul(p_stride),
        "presence buffer too small"
    );
    assert!(
        row_sign_byte_offsets.len() >= rows,
        "row_sign_byte_offsets buffer too small"
    );
    assert!(x.len() >= cols, "x buffer too small");
    assert!(y_out.len() >= rows, "y_out buffer too small");

    for r in 0..rows {
        let p_row = &presence[r * p_stride..(r + 1) * p_stride];
        let s_off = row_sign_byte_offsets[r] as usize;
        assert!(s_off <= signs.len(), "row sign offset out of bounds");
        y_out[r] = bitcos_dot_i8(cols, p_row, &signs[s_off..], &x[..cols]);
    }
}

// ---------------------------------------------------------------------------
// SPQT Base-243 (5 balanced trits per byte, 243..=255 rejected as sentinels)
// ---------------------------------------------------------------------------

pub const SPQT_RADIX: u16 = 243;
pub const SPQT_TRITS_PER_BYTE: usize = 5;
pub const SPQT_FIRST_SENTINEL: u8 = 243;
pub const SPQT_CORRUPT_SENTINEL: u8 = 0xF3;
pub const SPQT_ALL_NEG_BYTE: u8 = 0x00;
pub const SPQT_ALL_ZERO_BYTE: u8 = 0x79;
pub const SPQT_ALL_POS_BYTE: u8 = 0xF2;

const POW3_5: [u16; 5] = [1, 3, 9, 27, 81];

const fn build_spqt_lut() -> [[i8; 5]; 243] {
    let mut table = [[0i8; 5]; 243];
    let mut b = 0usize;
    while b < 243 {
        let mut rem = b;
        let mut k = 0usize;
        while k < 5 {
            table[b][k] = (rem % 3) as i8 - 1;
            rem /= 3;
            k += 1;
        }
        b += 1;
    }
    table
}

/// L1-resident `243 × 5 = 1,215` byte lookup table decoding valid Base-243 bytes with zero division.
pub const SPQT_LUT: [[i8; 5]; 243] = build_spqt_lut();

/// Pack 5 balanced [`Trit`]s into 1 Base-243 byte (`0..=242`).
#[inline]
pub const fn pack_spqt_5(trits: [Trit; 5]) -> u8 {
    let mut acc = 0u16;
    let mut k = 0usize;
    while k < 5 {
        acc += ((trits[k].as_i8() + 1) as u16) * POW3_5[k];
        k += 1;
    }
    acc as u8
}

/// Pack 5 `i8` values (normalized via [`Trit::from_i8`]) into 1 Base-243 byte.
#[inline]
pub const fn pack_spqt_i8_5(trits: [i8; 5]) -> u8 {
    pack_spqt_5([
        Trit::from_i8(trits[0]),
        Trit::from_i8(trits[1]),
        Trit::from_i8(trits[2]),
        Trit::from_i8(trits[3]),
        Trit::from_i8(trits[4]),
    ])
}

/// Unpack 1 Base-243 byte into 5 `i8` trits, returning `None` for sentinel bytes `>= 243`.
#[inline]
pub const fn unpack_spqt_i8_5(byte: u8) -> Option<[i8; 5]> {
    if byte >= SPQT_FIRST_SENTINEL {
        None
    } else {
        Some(SPQT_LUT[byte as usize])
    }
}

/// Unpack 1 Base-243 byte into 5 [`Trit`]s, returning `None` for sentinel bytes `>= 243`.
#[inline]
pub const fn unpack_spqt_5(byte: u8) -> Option<[Trit; 5]> {
    let Some(raw) = unpack_spqt_i8_5(byte) else {
        return None;
    };
    Some([
        Trit::from_i8(raw[0]),
        Trit::from_i8(raw[1]),
        Trit::from_i8(raw[2]),
        Trit::from_i8(raw[3]),
        Trit::from_i8(raw[4]),
    ])
}

/// Pack 64 balanced trits (`{-1, 0, +1}`) into 13 Base-243 bytes (`1.625` bpw w/ 1 tail zero pad).
#[inline]
pub const fn pack_spqt_64(trits: &[i8; 64]) -> [u8; 13] {
    let mut out = [SPQT_ALL_ZERO_BYTE; 13];
    let mut blk = 0usize;
    while blk < 12 {
        let base = blk * 5;
        out[blk] = pack_spqt_i8_5([
            trits[base],
            trits[base + 1],
            trits[base + 2],
            trits[base + 3],
            trits[base + 4],
        ]);
        blk += 1;
    }
    out[12] = pack_spqt_i8_5([trits[60], trits[61], trits[62], trits[63], 0]);
    out
}

/// Unpack 13 Base-243 bytes into 64 `i8` trits, returning `None` on any sentinel `>= 243`.
#[inline]
pub const fn unpack_spqt_64(packed: &[u8; 13]) -> Option<[i8; 64]> {
    let mut out = [0i8; 64];
    let mut blk = 0usize;
    while blk < 12 {
        let Some(lanes) = unpack_spqt_i8_5(packed[blk]) else {
            return None;
        };
        let base = blk * 5;
        out[base] = lanes[0];
        out[base + 1] = lanes[1];
        out[base + 2] = lanes[2];
        out[base + 3] = lanes[3];
        out[base + 4] = lanes[4];
        blk += 1;
    }
    let Some(tail) = unpack_spqt_i8_5(packed[12]) else {
        return None;
    };
    out[60] = tail[0];
    out[61] = tail[1];
    out[62] = tail[2];
    out[63] = tail[3];
    Some(out)
}

/// Exact 64-lane Base-243 SPQT dot product returning `None` on any corrupt sentinel (`>= 243`).
#[inline]
pub const fn spqt_dot_64_scalar(packed: &[u8; 13], activations: &[i8; 64]) -> Option<i32> {
    let mut acc = 0i32;
    let mut blk = 0usize;
    while blk < 12 {
        let Some(lanes) = unpack_spqt_i8_5(packed[blk]) else {
            return None;
        };
        let base = blk * 5;
        let mut k = 0usize;
        while k < 5 {
            acc += (lanes[k] as i32) * (activations[base + k] as i32);
            k += 1;
        }
        blk += 1;
    }
    let Some(tail) = unpack_spqt_i8_5(packed[12]) else {
        return None;
    };
    let mut k = 0usize;
    while k < 4 {
        acc += (tail[k] as i32) * (activations[60 + k] as i32);
        k += 1;
    }
    Some(acc)
}

/// Arbitrary-length L1-LUT Base-243 SPQT dot product, returning `None` on any sentinel (`>= 243`).
#[inline]
pub fn spqt_dot_scalar(packed: &[u8], activations: &[i8], cols: usize) -> Option<i32> {
    let needed_bytes = cols.div_ceil(5);
    assert!(packed.len() >= needed_bytes, "packed buffer too small");
    assert!(activations.len() >= cols, "activations buffer too small");

    let mut acc = 0i32;
    for (blk, &byte) in packed[..needed_bytes].iter().enumerate() {
        let lanes = unpack_spqt_i8_5(byte)?;
        let base = blk * 5;
        let rem = (cols - base).min(5);
        for k in 0..rem {
            acc += (lanes[k] as i32) * (activations[base + k] as i32);
        }
    }
    Some(acc)
}

// ---------------------------------------------------------------------------
// INT4 Symmetric Packed ([-7..=7] two's-complement, low nibble first) & GEMM
// ---------------------------------------------------------------------------

/// Pack two signed 4-bit integers (`lo` in `[3:0]`, `hi` in `[7:4]`) into 1 byte.
#[inline]
pub const fn int4_pack_byte(lo: i8, hi: i8) -> u8 {
    let lo_c = if lo < -8 {
        -8
    } else if lo > 7 {
        7
    } else {
        lo
    };
    let hi_c = if hi < -8 {
        -8
    } else if hi > 7 {
        7
    } else {
        hi
    };
    ((lo_c as u8) & 0x0F) | (((hi_c as u8) & 0x0F) << 4)
}

/// Pack symmetric `[-7..=7]` `i8` values into `packed_out` (2 nibbles/byte, low nibble first).
/// Odd tail lengths zero-pad the upper nibble. Returns the byte count written (`vals.len().div_ceil(2)`).
pub fn int4_pack_symmetric(vals: &[i8], packed_out: &mut [u8]) -> usize {
    let needed = vals.len().div_ceil(2);
    assert!(packed_out.len() >= needed, "packed_out buffer too small");
    for (b, dst) in packed_out[..needed].iter_mut().enumerate() {
        let idx = b * 2;
        let lo = vals[idx].clamp(-7, 7);
        let hi = if idx + 1 < vals.len() {
            vals[idx + 1].clamp(-7, 7)
        } else {
            0
        };
        *dst = ((lo as u8) & 0x0F) | (((hi as u8) & 0x0F) << 4);
    }
    needed
}

/// Unpack 1 packed INT4 byte into sign-extended `(lo, hi)` `i8` values in `[-8..=7]`.
#[inline]
pub const fn int4_unpack_byte(b: u8) -> (i8, i8) {
    let lo = ((b << 4) as i8) >> 4;
    let hi = (b as i8) >> 4;
    (lo, hi)
}

/// Exact scalar INT4 GEMM: `C[m x n] = A[m x k] * B^T[n x k]`,
/// where `a_packed` is `m x k.div_ceil(2)` and `b_t_packed` is `n x k.div_ceil(2)` row-major.
pub fn int4_gemm_scalar(
    a_packed: &[u8],
    b_t_packed: &[u8],
    c_out: &mut [i32],
    m: usize,
    n: usize,
    k: usize,
) {
    let k_bytes = k.div_ceil(2);
    assert!(
        a_packed.len() >= m.saturating_mul(k_bytes),
        "a_packed buffer too small"
    );
    assert!(
        b_t_packed.len() >= n.saturating_mul(k_bytes),
        "b_t_packed buffer too small"
    );
    assert!(
        c_out.len() >= m.saturating_mul(n),
        "c_out buffer too small"
    );

    let full_bytes = k >> 1;
    let has_tail = (k & 1) != 0;

    for r in 0..m {
        let a_row = &a_packed[r * k_bytes..(r + 1) * k_bytes];
        for col in 0..n {
            let b_row = &b_t_packed[col * k_bytes..(col + 1) * k_bytes];
            let mut acc = 0i32;
            for idx in 0..full_bytes {
                let (a0, a1) = int4_unpack_byte(a_row[idx]);
                let (b0, b1) = int4_unpack_byte(b_row[idx]);
                acc += (a0 as i32) * (b0 as i32) + (a1 as i32) * (b1 as i32);
            }
            if has_tail {
                let (a0, _) = int4_unpack_byte(a_row[full_bytes]);
                let (b0, _) = int4_unpack_byte(b_row[full_bytes]);
                acc += (a0 as i32) * (b0 as i32);
            }
            c_out[r * n + col] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// H-BitLinear: Pure-Integer WHT + INT4 Activations + BitNet b1.58 AbsMean
// ---------------------------------------------------------------------------

/// Pure-integer in-place Fast Walsh-Hadamard Transform (`H-BitLinear`, `arXiv:2504.18415`).
///
/// Returns `false` if `x` is empty or `x.len()` is not a power of two; otherwise performs
/// an unnormalized saturating Sylvester Hadamard butterfly (`WHT(WHT(x)) == N * x`).
pub fn hadamard_wht_inplace_i32(x: &mut [i32]) -> bool {
    let n = x.len();
    if n == 0 || !n.is_power_of_two() {
        return false;
    }
    let mut h = 1usize;
    while h < n {
        let step = h << 1;
        let mut i = 0usize;
        while i < n {
            let mut j = 0usize;
            while j < h {
                let a = x[i + j];
                let b = x[i + j + h];
                x[i + j] = a.saturating_add(b);
                x[i + j + h] = a.saturating_sub(b);
                j += 1;
            }
            i += step;
        }
        h = step;
    }
    true
}

/// Orthonormal Q15.16 Fast Walsh-Hadamard Transform (`WHT(x) / sqrt(N)`).
pub fn hadamard_wht_normalized_q16(x_q16: &mut [i32]) -> bool {
    if !hadamard_wht_inplace_i32(x_q16) {
        return false;
    }
    const Q16_INV_SQRT2: i64 = 46_341;
    let log2_n = x_q16.len().trailing_zeros();
    let shift = log2_n >> 1;
    let odd = (log2_n & 1) != 0;

    for slot in x_q16.iter_mut() {
        let v = *slot as i64;
        let norm = if odd {
            (v * Q16_INV_SQRT2) >> (16 + shift)
        } else {
            v >> shift
        };
        *slot = norm.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
    true
}

/// AbsMax symmetric INT4 activation quantizer (`[-7, +7]`) for `W1.58A4` `H-BitLinear`.
/// Returns AbsMax `scale_q16` (or `0` if all elements are zero).
pub fn quantize_activations_int4_symmetric(x_q16: &[i32], out_i4: &mut [i8]) -> i32 {
    assert_eq!(
        x_q16.len(),
        out_i4.len(),
        "x_q16 and out_i4 must match in length"
    );
    if x_q16.is_empty() {
        return 0;
    }

    let mut abs_max: u64 = 0;
    for &v in x_q16 {
        let a = (v as i64).unsigned_abs();
        if a > abs_max {
            abs_max = a;
        }
    }

    if abs_max == 0 {
        out_i4.fill(0);
        return 0;
    }

    let den = abs_max as u128;
    for (&x, dst) in x_q16.iter().zip(out_i4.iter_mut()) {
        let abs_x = (x as i64).unsigned_abs() as u128;
        let q_mag = ((7u128 * abs_x + (den >> 1)) / den).min(7) as i8;
        *dst = if x < 0 { -q_mag } else { q_mag };
    }

    if abs_max > (i32::MAX as u64) {
        i32::MAX
    } else {
        abs_max as i32
    }
}

/// Zero-alloc BitNet b1.58 AbsMean ternary weight quantizer (`arXiv:2504.12285`) in pure Q15.16.
///
/// Exact division-free thresholding: `|w_i| / gamma > 1/2` iff `2 * n * |w_i| > sum_j |w_j|`.
/// Returns exact `gamma` as a canonical `Ratio` in normalized Q15.16 units (`sum |w| / (n * 2^16)`).
pub fn quantize_weights_ternary_absmean_q16(weights_q16: &[i32], out_trits: &mut [Trit]) -> Ratio {
    assert_eq!(
        weights_q16.len(),
        out_trits.len(),
        "weights_q16 and out_trits must match in length"
    );
    let n = weights_q16.len();
    if n == 0 {
        return Ratio::ZERO;
    }

    let mut sum_abs: u128 = 0;
    for &w in weights_q16 {
        sum_abs += (w as i64).unsigned_abs() as u128;
    }
    if sum_abs == 0 {
        out_trits.fill(Trit::Zero);
        return Ratio::ZERO;
    }

    let two_n = (n as u128) * 2u128;
    for (&w, dst) in weights_q16.iter().zip(out_trits.iter_mut()) {
        let abs_w = (w as i64).unsigned_abs() as u128;
        *dst = if two_n * abs_w > sum_abs {
            if w > 0 { Trit::Pos } else { Trit::Neg }
        } else {
            Trit::Zero
        };
    }

    Ratio::from_u128(sum_abs, (n as u128) * (Q16_ONE as u128))
}
