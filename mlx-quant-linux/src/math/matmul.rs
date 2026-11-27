//! Real pure-integer / NEON / cross-arch quantized matrix kernels via `mlx-quant-asm`.

use crate::tensor::{DType, Tensor};
use core::fmt;

/// Errors returned by [`ComputeSingularity`] kernel validation and dispatch.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatmulError {
    UnsupportedDTypeCombination,
    InvalidShape,
    DimensionMismatch,
    BufferTooSmall,
    UnalignedOutput,
}

impl fmt::Display for MatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDTypeCombination => {
                f.write_str("unsupported tensor dtype combination for ComputeSingularity")
            }
            Self::InvalidShape => f.write_str("invalid tensor rank or dimensions"),
            Self::DimensionMismatch => f.write_str("matrix/vector inner or outer dimensions mismatch"),
            Self::BufferTooSmall => f.write_str("backing tensor buffer smaller than required"),
            Self::UnalignedOutput => f.write_str("output Int32 tensor buffer is unaligned or wrong dtype"),
        }
    }
}

impl std::error::Error for MatmulError {}

/// Executes real cross-architecture / AArch64 NEON SIMD matrix kernels with zero stubs.
pub struct ComputeSingularity;

impl ComputeSingularity {
    /// Execute exact quantized matrix multiplication or GEMV:
    /// - `(DType::Ternary2Bit, DType::Int8) -> DType::Int32`: `a` `[rows, cols]`, `b` `[cols]`, `c` `[rows]`
    /// - `(DType::Int4Packed, DType::Int4Packed) -> DType::Int32`: `a` `[m, k]`, `b` transposed `[n, k]`, `c` `[m, n]`
    pub fn execute_matmul(a: &Tensor, b: &Tensor, c: &mut Tensor) -> Result<(), MatmulError> {
        if c.dtype() != DType::Int32 {
            return Err(MatmulError::UnsupportedDTypeCombination);
        }

        match (a.dtype(), b.dtype()) {
            (DType::Ternary2Bit, DType::Int8) => {
                let a_shape = a.shape();
                let b_shape = b.shape();
                let c_shape = c.shape();
                if a_shape.ndim != 2 || b_shape.ndim != 1 || c_shape.ndim != 1 {
                    return Err(MatmulError::InvalidShape);
                }
                let rows = a_shape.dims[0];
                let cols = a_shape.dims[1];
                if b_shape.dims[0] != cols || c_shape.dims[0] != rows {
                    return Err(MatmulError::DimensionMismatch);
                }

                let needed_w = rows
                    .checked_mul(cols.div_ceil(4))
                    .ok_or(MatmulError::DimensionMismatch)?;
                let w_bytes = a.as_bytes();
                if w_bytes.len() < needed_w {
                    return Err(MatmulError::BufferTooSmall);
                }

                let x_i8 = b.as_i8_slice().ok_or(MatmulError::BufferTooSmall)?;
                if x_i8.len() < cols {
                    return Err(MatmulError::BufferTooSmall);
                }
                let y_i32 = c.as_i32_slice_mut().ok_or(MatmulError::UnalignedOutput)?;
                if y_i32.len() < rows {
                    return Err(MatmulError::BufferTooSmall);
                }

                mlx_quant_asm::ternary_2bit_gemv(
                    &w_bytes[..needed_w],
                    &x_i8[..cols],
                    &mut y_i32[..rows],
                    rows,
                    cols,
                );
                Ok(())
            }
            (DType::Int4Packed, DType::Int4Packed) => {
                let a_shape = a.shape();
                let b_shape = b.shape();
                let c_shape = c.shape();
                if a_shape.ndim != 2 || b_shape.ndim != 2 || c_shape.ndim != 2 {
                    return Err(MatmulError::InvalidShape);
                }
                let m = a_shape.dims[0];
                let k = a_shape.dims[1];
                let n = b_shape.dims[0];
                let b_k = b_shape.dims[1];
                if k != b_k || c_shape.dims[0] != m || c_shape.dims[1] != n {
                    return Err(MatmulError::DimensionMismatch);
                }

                let k_bytes = k.div_ceil(2);
                let needed_a = m
                    .checked_mul(k_bytes)
                    .ok_or(MatmulError::DimensionMismatch)?;
                let needed_b = n
                    .checked_mul(k_bytes)
                    .ok_or(MatmulError::DimensionMismatch)?;
                let needed_c = m.checked_mul(n).ok_or(MatmulError::DimensionMismatch)?;

                let a_bytes = a.as_bytes();
                let b_bytes = b.as_bytes();
                if a_bytes.len() < needed_a || b_bytes.len() < needed_b {
                    return Err(MatmulError::BufferTooSmall);
                }
                let c_i32 = c.as_i32_slice_mut().ok_or(MatmulError::UnalignedOutput)?;
                if c_i32.len() < needed_c {
                    return Err(MatmulError::BufferTooSmall);
                }

                mlx_quant_asm::int4_gemm(
                    &a_bytes[..needed_a],
                    &b_bytes[..needed_b],
                    &mut c_i32[..needed_c],
                    m,
                    n,
                    k,
                );
                Ok(())
            }
            _ => Err(MatmulError::UnsupportedDTypeCombination),
        }
    }

    /// Execute BITCOS sparse presence+sign ternary GEMV via `mlx_quant_asm::bitcos_gemv`.
    pub fn execute_bitcos_gemv(
        presence: &[u8],
        signs: &[u8],
        offsets: &[u32],
        x: &[i8],
        y_out: &mut [i32],
        rows: usize,
        cols: usize,
    ) -> Result<(), MatmulError> {
        let p_row_bytes = cols.div_ceil(8);
        let needed_p = rows
            .checked_mul(p_row_bytes)
            .ok_or(MatmulError::DimensionMismatch)?;
        if presence.len() < needed_p
            || offsets.len() < rows
            || x.len() < cols
            || y_out.len() < rows
        {
            return Err(MatmulError::BufferTooSmall);
        }

        let rem_bits = cols & 7;
        for (r, &off_u32) in offsets[..rows].iter().enumerate() {
            let off = off_u32 as usize;
            if off > signs.len() {
                return Err(MatmulError::BufferTooSmall);
            }
            if p_row_bytes > 0 {
                let row_p = &presence[r * p_row_bytes..(r + 1) * p_row_bytes];
                let mut nnz = 0usize;
                for (b_idx, &byte) in row_p.iter().enumerate() {
                    let masked = if b_idx + 1 == p_row_bytes && rem_bits != 0 {
                        byte & ((1u8 << rem_bits) - 1)
                    } else {
                        byte
                    };
                    nnz += masked.count_ones() as usize;
                }
                if off.saturating_add(nnz.div_ceil(8)) > signs.len() {
                    return Err(MatmulError::BufferTooSmall);
                }
            }
        }

        mlx_quant_asm::bitcos_gemv(presence, signs, offsets, x, y_out, rows, cols);
        Ok(())
    }

    /// Execute 16-element in-place NEON/scalar saturating Walsh-Hadamard transform (`H-BitLinear`).
    #[inline]
    pub fn execute_hadamard_wht_16(block: &mut [i32; 16]) {
        mlx_quant_asm::hadamard_wht_16(block);
    }
}
