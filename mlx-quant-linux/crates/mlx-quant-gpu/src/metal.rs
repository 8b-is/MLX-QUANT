//! Safe Rust wrapper around the macOS Metal runtime bridge (`ffi/metal_bridge.m`).
//!
//! Compiles cleanly in `#![no_std]` on non-macOS targets (e.g., `aarch64-unknown-none`),
//! returning [`GpuError::UnsupportedPlatform`] when `has_metal_bridge` is not configured.

use core::fmt;
use mlx_quant_core::beehive::TequilaConfig;

#[cfg(has_metal_bridge)]
use core::ffi::{c_char, c_int};

/// Errors returned by [`MetalRuntime`] initialization and kernel dispatch.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuError {
    UnsupportedPlatform,
    NoDevice,
    LibraryCompileFailed,
    PipelineCreateFailed,
    NotInitialized,
    BufferAllocationFailed,
    CommandEncodingFailed,
    ExecutionFailed,
    InvalidArgument,
    InvalidBufferLength,
    DimensionOverflow,
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => {
                f.write_str("Metal GPU bridge is only available on macOS targets")
            }
            Self::NoDevice => f.write_str("no Metal GPU device found on this host"),
            Self::LibraryCompileFailed => f.write_str("failed to load compiled Metal library"),
            Self::PipelineCreateFailed => f.write_str("failed to create Metal compute pipeline"),
            Self::NotInitialized => f.write_str("Metal bridge runtime not initialized"),
            Self::BufferAllocationFailed => f.write_str("failed to allocate Metal shared buffer"),
            Self::CommandEncodingFailed => f.write_str("failed to encode Metal compute command"),
            Self::ExecutionFailed => f.write_str("Metal command buffer execution failed"),
            Self::InvalidArgument => f.write_str("invalid argument passed to Metal bridge"),
            Self::InvalidBufferLength => f.write_str("slice buffer length does not match dimensions"),
            Self::DimensionOverflow => f.write_str("tensor dimensions exceed 32-bit GPU limits"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GpuError {}

#[cfg(has_metal_bridge)]
const METALLIB_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mlx_quant_beehive.metallib"));

#[cfg(has_metal_bridge)]
unsafe extern "C" {
    fn mq_metal_init(
        metallib_bytes: *const u8,
        metallib_len: usize,
        device_name_buf: *mut c_char,
        name_cap: usize,
    ) -> c_int;

    fn mq_metal_ternary_2bit_gemv(
        w_packed: *const u8,
        w_bytes: usize,
        x: *const i8,
        rows: u32,
        cols: u32,
        y_out: *mut i32,
        gpu_ns_out: *mut u64,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    fn mq_metal_bitcos_gemv(
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

    fn mq_metal_tequila_ste_q16(
        weights_q16: *mut i32,
        grads_q16: *const i32,
        len: u32,
        delta_q16: i32,
        lr_q16: i32,
        lambda_q16: i32,
        gpu_ns_out: *mut u64,
    ) -> c_int;
}

#[cfg(has_metal_bridge)]
#[inline]
const fn map_bridge_err(code: c_int) -> GpuError {
    match code {
        -1 => GpuError::InvalidArgument,
        -2 => GpuError::NoDevice,
        -3 => GpuError::LibraryCompileFailed,
        -4 => GpuError::PipelineCreateFailed,
        -5 => GpuError::NotInitialized,
        -6 => GpuError::BufferAllocationFailed,
        -7 => GpuError::CommandEncodingFailed,
        _ => GpuError::ExecutionFailed,
    }
}

/// Safe handle to the live Apple Metal GPU compute runtime.
#[derive(Clone, Debug)]
pub struct MetalRuntime {
    device_name_buf: [u8; 256],
    device_name_len: usize,
}

impl MetalRuntime {
    /// Initialize the Metal runtime, loading the compiled `mlx_quant_beehive.metallib`
    /// and caching compute pipeline states.
    pub fn new() -> Result<Self, GpuError> {
        #[cfg(has_metal_bridge)]
        {
            let mut device_name_buf = [0u8; 256];
            // SAFETY: `METALLIB_BYTES` is a valid static byte slice and `device_name_buf`
            // has writable capacity of 256 bytes.
            let rc = unsafe {
                mq_metal_init(
                    METALLIB_BYTES.as_ptr(),
                    METALLIB_BYTES.len(),
                    device_name_buf.as_mut_ptr().cast::<c_char>(),
                    device_name_buf.len(),
                )
            };
            if rc != 0 {
                return Err(map_bridge_err(rc));
            }
            let device_name_len = device_name_buf
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(device_name_buf.len());
            Ok(Self {
                device_name_buf,
                device_name_len,
            })
        }
        #[cfg(not(has_metal_bridge))]
        {
            Err(GpuError::UnsupportedPlatform)
        }
    }

    /// Name of the active Metal GPU device (e.g., `"Apple M1 Max"`).
    #[inline]
    pub fn device_name(&self) -> &str {
        core::str::from_utf8(&self.device_name_buf[..self.device_name_len]).unwrap_or("Apple Metal")
    }

    /// Execute pure-integer 2-bit ternary GEMV on Metal GPU and return elapsed GPU nanoseconds.
    ///
    /// Weight encoding is LSB-first `{0, 1, 2} -> {-1, 0, +1}` with row stride `cols.div_ceil(4)`.
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

        #[cfg(has_metal_bridge)]
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
            // SAFETY: Buffer lengths for `w_packed`, `x`, and `y_out` have been validated above.
            let rc = unsafe {
                mq_metal_ternary_2bit_gemv(
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
                return Err(map_bridge_err(rc));
            }
            Ok(gpu_ns)
        }

        #[cfg(not(has_metal_bridge))]
        {
            let _ = (rows_u32, cols_u32);
            Err(GpuError::UnsupportedPlatform)
        }
    }

    /// Execute pure-integer BITCOS sparse ternary GEMV on Metal GPU and return elapsed GPU nanoseconds.
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

        // Verify per-row sign byte offsets and required sign bit lengths fit inside `signs`.
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
                let need_sign_bytes = nnz.div_ceil(8);
                if off.saturating_add(need_sign_bytes) > signs.len() {
                    return Err(GpuError::InvalidBufferLength);
                }
            }
        }

        #[cfg(has_metal_bridge)]
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
            // SAFETY: All slice bounds and per-row sign byte ranges are verified above.
            let rc = unsafe {
                mq_metal_bitcos_gemv(
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
                return Err(map_bridge_err(rc));
            }
            Ok(gpu_ns)
        }

        #[cfg(not(has_metal_bridge))]
        {
            let _ = (rows_u32, cols_u32);
            Err(GpuError::UnsupportedPlatform)
        }
    }

    /// Execute Q15.16 Tequila deadzone reactivation STE update on Metal GPU.
    ///
    /// Returns elapsed GPU nanoseconds.
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

        #[cfg(has_metal_bridge)]
        {
            if weights_q16.is_empty() {
                return Ok(0);
            }
            let mut gpu_ns: u64 = 0;
            // SAFETY: `weights_q16` and `grads_q16` have identical non-zero length `len_u32`.
            let rc = unsafe {
                mq_metal_tequila_ste_q16(
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
                return Err(map_bridge_err(rc));
            }
            Ok(gpu_ns)
        }

        #[cfg(not(has_metal_bridge))]
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
