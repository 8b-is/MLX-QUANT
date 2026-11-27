//! Real fused `H-BitLinear` (`W1.58A4`) pure-integer forward kernel.
//!
//! Pipeline:
//! 1. `hadamard_wht_normalized_q16`: orthonormal Q15.16 Walsh-Hadamard transform (zero float).
//! 2. `quantize_activations_int4_symmetric`: symmetric AbsMax `[-7..=7]` INT4 quantization.
//! 3. `mlx_quant_asm::ternary_2bit_gemv`: AArch64 NEON / cross-arch 2-bit ternary GEMV -> `i32`.

use crate::ExecutionMode;
use crate::allocator::BumpAllocator;
use crate::tensor::{DType, Shape, Tensor};
use core::fmt;
use mlx_quant_core::ternary::{
    hadamard_wht_normalized_q16, quantize_activations_int4_symmetric,
};

/// Errors returned by [`FusionPolarity`] H-BitLinear execution.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FusionError {
    EmptyDimensions,
    NonPowerOfTwoCols,
    BufferTooSmall,
    InvalidTensorLayout,
}

impl fmt::Display for FusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimensions => f.write_str("H-BitLinear rows/cols must be non-zero"),
            Self::NonPowerOfTwoCols => {
                f.write_str("H-BitLinear Walsh-Hadamard transform requires power-of-two cols")
            }
            Self::BufferTooSmall => f.write_str("input, scratch, weight, or output buffer too small"),
            Self::InvalidTensorLayout => {
                f.write_str("invalid tensor dtype or shape for H-BitLinear forward")
            }
        }
    }
}

impl std::error::Error for FusionError {}

/// Telemetry emitted by [`FusionPolarity::h_bitlinear_forward`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FusionResult {
    pub absmax_scale_q16: i32,
    pub rows: usize,
    pub cols: usize,
    pub mode: ExecutionMode,
}

/// Fused `H-BitLinear` (`WHT -> AbsMax INT4 -> Ternary 2-bit GEMV`) compute operator.
pub struct FusionPolarity;

impl FusionPolarity {
    /// Execute fused `H-BitLinear` forward over `rows x cols` 2-bit packed ternary weights.
    ///
    /// Mutates `x_q16[..cols]` in-place with the normalized Walsh-Hadamard transform, quantizes
    /// into `act_i4_scratch[..cols]` in `[-7..=7]`, and computes exact `y_out[..rows] = W * q(WHT(x))`.
    pub fn h_bitlinear_forward(
        weights_2bit: &[u8],
        x_q16: &mut [i32],
        act_i4_scratch: &mut [i8],
        y_out: &mut [i32],
        rows: usize,
        cols: usize,
    ) -> Result<FusionResult, FusionError> {
        if rows == 0 || cols == 0 {
            return Err(FusionError::EmptyDimensions);
        }
        if !cols.is_power_of_two() {
            return Err(FusionError::NonPowerOfTwoCols);
        }
        let needed_w = rows
            .checked_mul(cols.div_ceil(4))
            .ok_or(FusionError::BufferTooSmall)?;
        if weights_2bit.len() < needed_w
            || x_q16.len() < cols
            || act_i4_scratch.len() < cols
            || y_out.len() < rows
        {
            return Err(FusionError::BufferTooSmall);
        }

        if !hadamard_wht_normalized_q16(&mut x_q16[..cols]) {
            return Err(FusionError::NonPowerOfTwoCols);
        }
        let absmax_scale_q16 =
            quantize_activations_int4_symmetric(&x_q16[..cols], &mut act_i4_scratch[..cols]);
        mlx_quant_asm::ternary_2bit_gemv(
            &weights_2bit[..needed_w],
            &act_i4_scratch[..cols],
            &mut y_out[..rows],
            rows,
            cols,
        );

        Ok(FusionResult {
            absmax_scale_q16,
            rows,
            cols,
            mode: ExecutionMode::Measured,
        })
    }

    /// Execute a single-row fused `H-BitLinear` dot product returning `(dot_i32, absmax_scale_q16)`.
    pub fn h_bitlinear_dot_row(
        weights_2bit_row: &[u8],
        x_q16: &mut [i32],
        act_i4_scratch: &mut [i8],
    ) -> Result<(i32, i32), FusionError> {
        let mut y = [0i32; 1];
        let cols = x_q16.len();
        let res = Self::h_bitlinear_forward(
            weights_2bit_row,
            x_q16,
            act_i4_scratch,
            &mut y,
            1,
            cols,
        )?;
        Ok((y[0], res.absmax_scale_q16))
    }

    /// Arena-backed fused `H-BitLinear` forward returning an allocated `DType::Int32` `[rows]` [`Tensor`].
    pub fn h_bitlinear_forward_tensor<'a>(
        weights: &Tensor<'a>,
        x_q16: &mut [i32],
        allocator: &'a BumpAllocator,
    ) -> Result<(Tensor<'a>, FusionResult), FusionError> {
        if weights.dtype() != DType::Ternary2Bit || weights.shape().ndim != 2 {
            return Err(FusionError::InvalidTensorLayout);
        }
        let rows = weights.shape().dims[0];
        let cols = weights.shape().dims[1];
        if x_q16.len() != cols {
            return Err(FusionError::BufferTooSmall);
        }

        let act_scratch = allocator
            .alloc_slice_mut::<i8>(cols)
            .ok_or(FusionError::BufferTooSmall)?;
        let mut out_tensor = Tensor::new(Shape::new_1d(rows), DType::Int32, allocator)
            .ok_or(FusionError::BufferTooSmall)?;
        let y_out = out_tensor
            .as_i32_slice_mut()
            .ok_or(FusionError::InvalidTensorLayout)?;

        let result = Self::h_bitlinear_forward(
            weights.as_bytes(),
            x_q16,
            act_scratch,
            y_out,
            rows,
            cols,
        )?;
        Ok((out_tensor, result))
    }
}
