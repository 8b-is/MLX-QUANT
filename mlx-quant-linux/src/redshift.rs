//! Pure-integer quantization ("redshifting") backed by `mlx_quant_core::ternary`.
//!
//! Converts Q15.16 weights/activations into MLX 2-bit ternary, symmetric INT4 packed,
//! and sparse BITCOS representations with zero floating-point arithmetic.

use crate::allocator::BumpAllocator;
use crate::tensor::{DType, Shape, Tensor};
use mlx_quant_core::ratio::{Q16_ONE, Ratio};
use mlx_quant_core::ternary::{
    Trit, bitcos_effective_bpw, bitcos_pack_into, int4_pack_byte, pack_mlx_ternary_4,
};

/// Result of packing balanced trits into a [`DType::BitcosSparse`] tensor.
#[derive(Debug)]
pub struct BitcosTensor<'a> {
    pub tensor: Tensor<'a>,
    pub presence_bytes: usize,
    pub sign_bytes: usize,
    pub nnz: usize,
    pub effective_bpw: Ratio,
}

/// Pure-integer quantizer routing tensors into lower-bitwidth representations.
pub struct Redshift;

impl Redshift {
    /// Fallible BitNet b1.58 AbsMean ternary quantization of Q15.16 weights into a shaped
    /// [`DType::Ternary2Bit`] [`Tensor`] (`{0,1,2} -> {-1,0,+1}` LSB-first), returning
    /// `(tensor, absmean_gamma_q16)`.
    pub fn try_redshift_q16_to_ternary_2bit<'a>(
        w_q16: &[i32],
        shape: Shape,
        allocator: &'a BumpAllocator,
    ) -> Option<(Tensor<'a>, Ratio)> {
        let numel = shape.checked_numel()?;
        if numel != w_q16.len() {
            return None;
        }
        let mut tensor = Tensor::new(shape, DType::Ternary2Bit, allocator)?;
        if numel == 0 {
            return Some((tensor, Ratio::ZERO));
        }

        let mut sum_abs: u128 = 0;
        for &w in w_q16 {
            sum_abs += (w as i64).unsigned_abs() as u128;
        }
        if sum_abs == 0 {
            // Canonical 2-bit MLX code for all zeros is `0b01` per trit (`0x55` per byte).
            tensor.as_bytes_mut().fill(0x55);
            return Some((tensor, Ratio::ZERO));
        }

        let two_n = (numel as u128) * 2u128;
        let last_dim = if shape.ndim <= 1 {
            numel
        } else {
            shape.dims[shape.ndim as usize - 1]
        };
        let outer = numel.checked_div(last_dim).unwrap_or(0);
        let row_bytes = last_dim.div_ceil(4);
        let out = tensor.as_bytes_mut();

        for r in 0..outer {
            let row_w = &w_q16[r * last_dim..(r + 1) * last_dim];
            let row_out = &mut out[r * row_bytes..(r + 1) * row_bytes];
            let trit_at = |idx: usize| -> Trit {
                if idx >= last_dim {
                    return Trit::Zero;
                }
                let w = row_w[idx];
                let abs_w = (w as i64).unsigned_abs() as u128;
                if two_n * abs_w > sum_abs {
                    if w > 0 { Trit::Pos } else { Trit::Neg }
                } else {
                    Trit::Zero
                }
            };
            for (b, dst) in row_out.iter_mut().enumerate() {
                let base = b * 4;
                *dst = pack_mlx_ternary_4([
                    trit_at(base),
                    trit_at(base + 1),
                    trit_at(base + 2),
                    trit_at(base + 3),
                ]);
            }
        }

        let gamma = Ratio::from_u128(sum_abs, (numel as u128) * (Q16_ONE as u128));
        Some((tensor, gamma))
    }

    /// Quantize 1D Q15.16 weights into a [`DType::Ternary2Bit`] [`Tensor`] and exact AbsMean `Ratio`.
    pub fn redshift_q16_to_ternary_2bit<'a>(
        w_q16: &[i32],
        allocator: &'a BumpAllocator,
    ) -> (Tensor<'a>, Ratio) {
        Self::try_redshift_q16_to_ternary_2bit(w_q16, Shape::new_1d(w_q16.len()), allocator)
            .expect("BumpAllocator exhausted in redshift_q16_to_ternary_2bit")
    }

    /// Fallible symmetric AbsMax INT4 quantization (`[-7..=7]`, 2 nibbles/byte) for shaped Q15.16 inputs.
    /// Returns `(tensor, absmax_scale_q16)`.
    pub fn try_redshift_q16_to_int4<'a>(
        x_q16: &[i32],
        shape: Shape,
        allocator: &'a BumpAllocator,
    ) -> Option<(Tensor<'a>, i32)> {
        let numel = shape.checked_numel()?;
        if numel != x_q16.len() {
            return None;
        }
        let mut tensor = Tensor::new(shape, DType::Int4Packed, allocator)?;
        if numel == 0 {
            return Some((tensor, 0));
        }

        let mut abs_max: u64 = 0;
        for &v in x_q16 {
            let a = (v as i64).unsigned_abs();
            if a > abs_max {
                abs_max = a;
            }
        }
        if abs_max == 0 {
            tensor.zero();
            return Some((tensor, 0));
        }

        let den = abs_max as u128;
        let quant = |x: i32| -> i8 {
            let abs_x = (x as i64).unsigned_abs() as u128;
            let q_mag = ((7u128 * abs_x + (den >> 1)) / den).min(7) as i8;
            if x < 0 { -q_mag } else { q_mag }
        };

        let last_dim = if shape.ndim <= 1 {
            numel
        } else {
            shape.dims[shape.ndim as usize - 1]
        };
        let outer = numel.checked_div(last_dim).unwrap_or(0);
        let row_bytes = last_dim.div_ceil(2);
        let out = tensor.as_bytes_mut();

        for r in 0..outer {
            let row_x = &x_q16[r * last_dim..(r + 1) * last_dim];
            let row_out = &mut out[r * row_bytes..(r + 1) * row_bytes];
            for (b, dst) in row_out.iter_mut().enumerate() {
                let idx = b * 2;
                let lo = quant(row_x[idx]);
                let hi = if idx + 1 < last_dim {
                    quant(row_x[idx + 1])
                } else {
                    0
                };
                *dst = int4_pack_byte(lo, hi);
            }
        }

        let scale_q16 = if abs_max > (i32::MAX as u64) {
            i32::MAX
        } else {
            abs_max as i32
        };
        Some((tensor, scale_q16))
    }

    /// Quantize 1D Q15.16 activations into [`DType::Int4Packed`] and return `(tensor, absmax_scale_q16)`.
    pub fn redshift_q16_to_int4<'a>(
        x_q16: &[i32],
        allocator: &'a BumpAllocator,
    ) -> (Tensor<'a>, i32) {
        Self::try_redshift_q16_to_int4(x_q16, Shape::new_1d(x_q16.len()), allocator)
            .expect("BumpAllocator exhausted in redshift_q16_to_int4")
    }

    /// Fallible BITCOS sparse packing (`presence` bitmap + compacted `signs` stream) into a
    /// [`DType::BitcosSparse`] [`Tensor`] allocated from `allocator`.
    pub fn try_redshift_trits_to_bitcos<'a>(
        trits: &[Trit],
        allocator: &'a BumpAllocator,
    ) -> Option<BitcosTensor<'a>> {
        let mut tensor = Tensor::new(Shape::new_1d(trits.len()), DType::BitcosSparse, allocator)?;
        let p_bytes = trits.len().div_ceil(8);
        let (presence_out, signs_out) = tensor.as_bytes_mut().split_at_mut(p_bytes);
        let (nnz, sign_bytes) = bitcos_pack_into(trits, presence_out, signs_out);
        let effective_bpw = bitcos_effective_bpw(trits.len(), nnz);
        Some(BitcosTensor {
            tensor,
            presence_bytes: p_bytes,
            sign_bytes,
            nnz,
            effective_bpw,
        })
    }

    /// Pack balanced trits into a [`BitcosTensor`] (`DType::BitcosSparse`).
    pub fn redshift_trits_to_bitcos<'a>(
        trits: &[Trit],
        allocator: &'a BumpAllocator,
    ) -> BitcosTensor<'a> {
        Self::try_redshift_trits_to_bitcos(trits, allocator)
            .expect("BumpAllocator exhausted in redshift_trits_to_bitcos")
    }
}
