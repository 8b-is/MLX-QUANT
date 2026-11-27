//! Safe Rust wrapper around the NVIDIA CUDA runtime bridge (`ffi/cuda_bridge.cu`).
//!
//! Compiles cleanly in `#![no_std]` and on non-CUDA targets, returning
//! [`GpuError::UnsupportedPlatform`] when `has_cuda_bridge` is not configured.

use crate::metal::GpuError;
use mlx_quant_core::beehive::TequilaConfig;

#[cfg(has_cuda_bridge)]
use core::ffi::{c_char, c_int};

#[cfg(has_cuda_bridge)]
unsafe extern "C" {
    fn mq_cuda_init(
        device_name_buf: *mut c_char,
        name_cap: usize,
        cc_major_out: *mut c_int,
        cc_minor_out: *mut c_int,
    ) -> c_int;

    fn mq_cuda_ternary_2bit_gemv(
        w_packed: *const u8,
        w_bytes: usize,
        x: *const i8,
        rows: u32,
        cols: u32,
        y_out: *mut i32,
        gpu_ns_out: *mut u64,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    fn mq_cuda_bitcos_gemv(
        presence: *const u8,
        p_bytes: usize,
        signs: *const u8,
        s_bytes: usize,
        row_sign_byte_offsets: *const u32,
        x: *const i8,
        rows: u32,
        cols: u32,
        y_out: *mut i32,
        gpu_ns_out: *mut u64,
    ) -> c_int;

    fn mq_cuda_tequila_ste_q16(
        weights_q16: *mut i32,
        grads_q16: *const i32,
        len: u32,
        delta_q16: i32,
        lr_q16: i32,
        lambda_q16: i32,
        gpu_ns_out: *mut u64,
    ) -> c_int;
}

#[cfg(has_cuda_bridge)]
#[inline]
const fn map_cuda_err(code: c_int) -> GpuError {
    match code {
        -1 => GpuError::InvalidArgument,
        -2 => GpuError::NoDevice,
        -5 => GpuError::NotInitialized,
        -6 => GpuError::BufferAllocationFailed,
        -7 => GpuError::CommandEncodingFailed,
        _ => GpuError::ExecutionFailed,
    }
}

/// Safe handle to the live NVIDIA CUDA GPU compute runtime (e.g., RTX 6000 Ada `SM_89`).
#[derive(Clone, Debug)]
pub struct CudaRuntime {
    device_name_buf: [u8; 256],
    device_name_len: usize,
    cc_major: i32,
    cc_minor: i32,
}

impl CudaRuntime {
    /// Initialize CUDA device 0 and query its device name and compute capability (`SM_xy`).
    pub fn new() -> Result<Self, GpuError> {
        #[cfg(has_cuda_bridge)]
        {
            let mut device_name_buf = [0u8; 256];
            let mut cc_major: c_int = 0;
            let mut cc_minor: c_int = 0;
            // SAFETY: Buffers are valid writable storage passed by length to `mq_cuda_init`.
            let rc = unsafe {
                mq_cuda_init(
                    device_name_buf.as_mut_ptr().cast::<c_char>(),
                    device_name_buf.len(),
                    &mut cc_major,
                    &mut cc_minor,
                )
            };
            if rc != 0 {
                return Err(map_cuda_err(rc));
            }
            let device_name_len = device_name_buf
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(device_name_buf.len());
            Ok(Self {
                device_name_buf,
                device_name_len,
                cc_major,
                cc_minor,
            })
        }
        #[cfg(not(has_cuda_bridge))]
        {
            Err(GpuError::UnsupportedPlatform)
        }
    }

    /// Active CUDA GPU device name (e.g., `"NVIDIA RTX 6000 Ada Generation"`).
    #[inline]
    pub fn device_name(&self) -> &str {
        core::str::from_utf8(&self.device_name_buf[..self.device_name_len]).unwrap_or("NVIDIA CUDA")
    }

    /// CUDA compute capability `(major, minor)` (e.g., `(8, 9)` on RTX 6000 Ada).
    #[inline]
    pub const fn compute_capability(&self) -> (i32, i32) {
        (self.cc_major, self.cc_minor)
    }

    /// Execute pure-integer 2-bit ternary GEMV on CUDA GPU and return elapsed GPU nanoseconds.
    pub fn ternary_2bit_gemv(
        &self,
        w_packed: &[u8],
        x: &[i8],
        y_out: &mut [i32],
        rows: usize,
        cols: usize,
    ) -> Result<u64, GpuError> {
        let rows_u32 = u32::try_from(rows).map_err(|_| GpuError::DimensionOverflow)?;
        let cols_u32 = u32::try_from(cols).map_err(|_| GpuError::DimensionOverflow)?;
        let row_bytes = cols.div_ceil(4);
        let expected_w_bytes = rows
            .checked_mul(row_bytes)
            .ok_or(GpuError::DimensionOverflow)?;

        if w_packed.len() < expected_w_bytes || x.len() < cols || y_out.len() < rows {
            return Err(GpuError::InvalidBufferLength);
        }

        #[cfg(has_cuda_bridge)]
        {
            if rows == 0 {
                return Ok(0);
            }
            let mut gpu_ns: u64 = 0;
            let w_ptr = if w_packed.is_empty() {
                core::ptr::null()
            } else {
                w_packed.as_ptr()
            };
            let x_ptr = if x.is_empty() {
                core::ptr::null()
            } else {
                x.as_ptr()
            };
            // SAFETY: All slice bounds are verified above.
            let rc = unsafe {
                mq_cuda_ternary_2bit_gemv(
                    w_ptr,
                    w_packed.len(),
                    x_ptr,
                    rows_u32,
                    cols_u32,
                    y_out.as_mut_ptr(),
                    &mut gpu_ns,
                )
            };
            if rc != 0 {
                return Err(map_cuda_err(rc));
            }
            Ok(gpu_ns)
        }

        #[cfg(not(has_cuda_bridge))]
        {
            let _ = (rows_u32, cols_u32);
            Err(GpuError::UnsupportedPlatform)
        }
    }

    /// Execute pure-integer BITCOS sparse ternary GEMV on CUDA GPU and return elapsed GPU nanoseconds.
    #[allow(clippy::too_many_arguments)]
    pub fn bitcos_gemv(
        &self,
        presence: &[u8],
        signs: &[u8],
        row_sign_byte_offsets: &[u32],
        x: &[i8],
        y_out: &mut [i32],
        rows: usize,
        cols: usize,
    ) -> Result<u64, GpuError> {
        let rows_u32 = u32::try_from(rows).map_err(|_| GpuError::DimensionOverflow)?;
        let cols_u32 = u32::try_from(cols).map_err(|_| GpuError::DimensionOverflow)?;
        let p_row_bytes = cols.div_ceil(8);
        let expected_p_bytes = rows
            .checked_mul(p_row_bytes)
            .ok_or(GpuError::DimensionOverflow)?;

        if presence.len() < expected_p_bytes
            || row_sign_byte_offsets.len() < rows
            || x.len() < cols
            || y_out.len() < rows
        {
            return Err(GpuError::InvalidBufferLength);
        }

        let rem_bits = cols & 7;
        for (r, &offset) in row_sign_byte_offsets[..rows].iter().enumerate() {
            let off = offset as usize;
            if off > signs.len() {
                return Err(GpuError::InvalidBufferLength);
            }
            if p_row_bytes > 0 {
                let row_pres = &presence[r * p_row_bytes..(r + 1) * p_row_bytes];
                let mut nnz = 0usize;
                for (b_idx, &byte) in row_pres.iter().enumerate() {
                    let masked = if b_idx + 1 == p_row_bytes && rem_bits != 0 {
                        byte & ((1u8 << rem_bits) - 1)
                    } else {
                        byte
                    };
                    nnz += masked.count_ones() as usize;
                }
                if off.saturating_add(nnz.div_ceil(8)) > signs.len() {
                    return Err(GpuError::InvalidBufferLength);
                }
            }
        }

        #[cfg(has_cuda_bridge)]
        {
            if rows == 0 {
                return Ok(0);
            }
            let mut gpu_ns: u64 = 0;
            let p_ptr = if presence.is_empty() {
                core::ptr::null()
            } else {
                presence.as_ptr()
            };
            let s_ptr = if signs.is_empty() {
                core::ptr::null()
            } else {
                signs.as_ptr()
            };
            let x_ptr = if x.is_empty() {
                core::ptr::null()
            } else {
                x.as_ptr()
            };
            // SAFETY: Bounds and per-row sign offsets validated above.
            let rc = unsafe {
                mq_cuda_bitcos_gemv(
                    p_ptr,
                    presence.len(),
                    s_ptr,
                    signs.len(),
                    row_sign_byte_offsets.as_ptr(),
                    x_ptr,
                    rows_u32,
                    cols_u32,
                    y_out.as_mut_ptr(),
                    &mut gpu_ns,
                )
            };
            if rc != 0 {
                return Err(map_cuda_err(rc));
            }
            Ok(gpu_ns)
        }

        #[cfg(not(has_cuda_bridge))]
        {
            let _ = (rows_u32, cols_u32);
            Err(GpuError::UnsupportedPlatform)
        }
    }

    /// Execute Q15.16 Tequila deadzone reactivation STE update on CUDA GPU.
    pub fn tequila_ste_q16(
        &self,
        weights_q16: &mut [i32],
        grads_q16: &[i32],
        delta_q16: i32,
        lr_q16: i32,
        lambda_q16: i32,
    ) -> Result<u64, GpuError> {
        if weights_q16.len() != grads_q16.len() {
            return Err(GpuError::InvalidBufferLength);
        }
        let len_u32 = u32::try_from(weights_q16.len()).map_err(|_| GpuError::DimensionOverflow)?;

        #[cfg(has_cuda_bridge)]
        {
            if weights_q16.is_empty() {
                return Ok(0);
            }
            let mut gpu_ns: u64 = 0;
            // SAFETY: Equal non-zero slice lengths verified above.
            let rc = unsafe {
                mq_cuda_tequila_ste_q16(
                    weights_q16.as_mut_ptr(),
                    grads_q16.as_ptr(),
                    len_u32,
                    delta_q16,
                    lr_q16,
                    lambda_q16,
                    &mut gpu_ns,
                )
            };
            if rc != 0 {
                return Err(map_cuda_err(rc));
            }
            Ok(gpu_ns)
        }

        #[cfg(not(has_cuda_bridge))]
        {
            let _ = (len_u32, delta_q16, lr_q16, lambda_q16);
            Err(GpuError::UnsupportedPlatform)
        }
    }

    /// Execute Q15.16 Tequila deadzone reactivation STE update using a [`TequilaConfig`].
    #[inline]
    pub fn tequila_ste_with_config(
        &self,
        weights_q16: &mut [i32],
        grads_q16: &[i32],
        cfg: TequilaConfig,
    ) -> Result<u64, GpuError> {
        self.tequila_ste_q16(
            weights_q16,
            grads_q16,
            cfg.delta_q16,
            cfg.lr_q16,
            cfg.lambda_q16,
        )
    }
}
