//! Safe GPU dispatch doorbell wrapping `mlx_quant_gpu::MetalRuntime` (macOS) and
//! `mlx_quant_gpu::CudaRuntime` (Linux/NVIDIA CUDA, e.g. RTX 6000 Ada) with a validated
//! portable CPU ring simulator fallback when no native GPU runtime is present.

use crate::ExecutionMode;
use crate::pulsar::Pulsar;
use crate::tensor::{DType, Tensor};
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering, fence};
use mlx_quant_core::beehive::{TequilaConfig, tequila_ste_step_q16};
use mlx_quant_core::ternary::{bitcos_gemv_scalar, ternary_2bit_gemv_scalar};
use mlx_quant_gpu::{CudaRuntime, GpuError, MetalRuntime};

/// Hardware reference base MMIO address for Apple Silicon AGX (informational descriptor only).
pub const AGX_MMIO_BASE: usize = 0x2_8000_0000;
/// Hardware reference doorbell offset within AGX MMIO space (informational descriptor only).
pub const AGX_DOORBELL_OFFSET: usize = 0x4000;

/// Safe GPU submission doorbell with live Apple Metal / NVIDIA CUDA support and CPU simulator fallback.
#[derive(Debug)]
pub struct AgxDoorbell {
    metal: Option<MetalRuntime>,
    cuda: Option<CudaRuntime>,
    ring_counter: AtomicU64,
    last_gpu_ns: AtomicU64,
    last_ring_id: AtomicU32,
}

impl Default for AgxDoorbell {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl AgxDoorbell {
    /// Initialize `AgxDoorbell`, attaching live `MetalRuntime` on macOS or live `CudaRuntime`
    /// on Linux+CUDA hosts, cleanly falling back to `"Portable Ring Simulator"` otherwise.
    pub fn new() -> Self {
        let metal = MetalRuntime::new().ok();
        let cuda = if metal.is_none() {
            CudaRuntime::new().ok()
        } else {
            None
        };
        Self {
            metal,
            cuda,
            ring_counter: AtomicU64::new(0),
            last_gpu_ns: AtomicU64::new(0),
            last_ring_id: AtomicU32::new(0),
        }
    }

    /// Construct a deterministic CPU-only simulator doorbell for parity and fallback tests.
    pub fn new_simulator() -> Self {
        Self {
            metal: None,
            cuda: None,
            ring_counter: AtomicU64::new(0),
            last_gpu_ns: AtomicU64::new(0),
            last_ring_id: AtomicU32::new(0),
        }
    }

    /// Honest backend mode string: `"Apple Metal Live GPU"`, `"NVIDIA CUDA Live GPU"`,
    /// or `"Portable Ring Simulator"`.
    #[inline]
    pub fn backend_mode(&self) -> &'static str {
        if self.metal.is_some() {
            "Apple Metal Live GPU"
        } else if self.cuda.is_some() {
            "NVIDIA CUDA Live GPU"
        } else {
            "Portable Ring Simulator"
        }
    }

    /// Execution mode classification for audit reporting.
    #[inline]
    pub fn execution_mode(&self) -> ExecutionMode {
        if self.is_live_gpu() {
            ExecutionMode::Measured
        } else {
            ExecutionMode::DesignedDescriptor
        }
    }

    /// Active GPU device name, or `"Portable Ring Simulator"` when running on CPU fallback.
    #[inline]
    pub fn device_name(&self) -> &str {
        if let Some(rt) = &self.metal {
            rt.device_name()
        } else if let Some(rt) = &self.cuda {
            rt.device_name()
        } else {
            "Portable Ring Simulator"
        }
    }

    /// Returns `true` when backed by a live Apple Metal or NVIDIA CUDA GPU runtime.
    #[inline]
    pub fn is_live_gpu(&self) -> bool {
        self.metal.is_some() || self.cuda.is_some()
    }

    /// Monotonic count of doorbell rings and GPU/simulator kernel submissions.
    #[inline]
    pub fn ring_counter(&self) -> u64 {
        self.ring_counter.load(Ordering::Acquire)
    }

    /// Elapsed kernel nanoseconds recorded by the most recent submission.
    #[inline]
    pub fn last_gpu_ns(&self) -> u64 {
        self.last_gpu_ns.load(Ordering::Acquire)
    }

    /// Most recently signaled ring ID.
    #[inline]
    pub fn last_ring_id(&self) -> u32 {
        self.last_ring_id.load(Ordering::Acquire)
    }

    /// Signal a release-fenced queue doorbell event, returning the updated monotonic counter.
    #[inline]
    pub fn ring(&self, ring_id: u32) -> u64 {
        fence(Ordering::Release);
        self.last_ring_id.store(ring_id, Ordering::Release);
        self.ring_counter.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Submit a 2-bit packed ternary GEMV (`y_out = W * x`), returning elapsed kernel nanoseconds.
    pub fn submit_ternary_2bit_gemv(
        &self,
        w_packed: &[u8],
        x: &[i8],
        y_out: &mut [i32],
        rows: usize,
        cols: usize,
    ) -> Result<u64, GpuError> {
        let ns = if let Some(rt) = &self.metal {
            rt.ternary_2bit_gemv(w_packed, x, y_out, rows, cols)?
        } else if let Some(rt) = &self.cuda {
            rt.ternary_2bit_gemv(w_packed, x, y_out, rows, cols)?
        } else {
            let _ = u32::try_from(rows).map_err(|_| GpuError::DimensionOverflow)?;
            let _ = u32::try_from(cols).map_err(|_| GpuError::DimensionOverflow)?;
            let stride = cols.div_ceil(4);
            let expected_w = rows
                .checked_mul(stride)
                .ok_or(GpuError::DimensionOverflow)?;
            if w_packed.len() < expected_w || x.len() < cols || y_out.len() < rows {
                return Err(GpuError::InvalidBufferLength);
            }
            let t0 = Pulsar::now_ticks();
            ternary_2bit_gemv_scalar(w_packed, x, y_out, rows, cols);
            let t1 = Pulsar::now_ticks();
            Pulsar::elapsed_ns(t0, t1)
        };
        self.last_gpu_ns.store(ns, Ordering::Release);
        self.ring_counter.fetch_add(1, Ordering::AcqRel);
        Ok(ns)
    }

    /// Submit a BITCOS sparse ternary GEMV, returning elapsed kernel nanoseconds.
    #[allow(clippy::too_many_arguments)]
    pub fn submit_bitcos_gemv(
        &self,
        presence: &[u8],
        signs: &[u8],
        row_sign_byte_offsets: &[u32],
        x: &[i8],
        y_out: &mut [i32],
        rows: usize,
        cols: usize,
    ) -> Result<u64, GpuError> {
        let ns = if let Some(rt) = &self.metal {
            rt.bitcos_gemv(
                presence,
                signs,
                row_sign_byte_offsets,
                x,
                y_out,
                rows,
                cols,
            )?
        } else if let Some(rt) = &self.cuda {
            rt.bitcos_gemv(
                presence,
                signs,
                row_sign_byte_offsets,
                x,
                y_out,
                rows,
                cols,
            )?
        } else {
            let _ = u32::try_from(rows).map_err(|_| GpuError::DimensionOverflow)?;
            let _ = u32::try_from(cols).map_err(|_| GpuError::DimensionOverflow)?;
            let p_row_bytes = cols.div_ceil(8);
            let expected_p = rows
                .checked_mul(p_row_bytes)
                .ok_or(GpuError::DimensionOverflow)?;
            if presence.len() < expected_p
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
            let t0 = Pulsar::now_ticks();
            bitcos_gemv_scalar(
                presence,
                signs,
                row_sign_byte_offsets,
                x,
                y_out,
                rows,
                cols,
            );
            let t1 = Pulsar::now_ticks();
            Pulsar::elapsed_ns(t0, t1)
        };
        self.last_gpu_ns.store(ns, Ordering::Release);
        self.ring_counter.fetch_add(1, Ordering::AcqRel);
        Ok(ns)
    }

    /// Submit a Q15.16 Tequila deadzone reactivation STE update, returning elapsed kernel nanoseconds.
    pub fn submit_tequila_ste_q16(
        &self,
        weights_q16: &mut [i32],
        grads_q16: &[i32],
        cfg: TequilaConfig,
    ) -> Result<u64, GpuError> {
        let ns = if let Some(rt) = &self.metal {
            rt.tequila_ste_with_config(weights_q16, grads_q16, cfg)?
        } else if let Some(rt) = &self.cuda {
            rt.tequila_ste_with_config(weights_q16, grads_q16, cfg)?
        } else {
            if weights_q16.len() != grads_q16.len() {
                return Err(GpuError::InvalidBufferLength);
            }
            let _ = u32::try_from(weights_q16.len()).map_err(|_| GpuError::DimensionOverflow)?;
            let t0 = Pulsar::now_ticks();
            tequila_ste_step_q16(weights_q16, grads_q16, cfg);
            let t1 = Pulsar::now_ticks();
            Pulsar::elapsed_ns(t0, t1)
        };
        self.last_gpu_ns.store(ns, Ordering::Release);
        self.ring_counter.fetch_add(1, Ordering::AcqRel);
        Ok(ns)
    }

    /// Submit a `Tensor`-backed 2-bit ternary GEMV (`[rows, cols] * [cols] -> [rows]`).
    pub fn submit_tensor_gemv(
        &self,
        w: &Tensor<'_>,
        x: &Tensor<'_>,
        y: &mut Tensor<'_>,
    ) -> Result<u64, GpuError> {
        if w.dtype() != DType::Ternary2Bit || x.dtype() != DType::Int8 || y.dtype() != DType::Int32
        {
            return Err(GpuError::InvalidArgument);
        }
        let w_shape = w.shape();
        let x_shape = x.shape();
        let y_shape = y.shape();
        if w_shape.ndim != 2 || x_shape.ndim != 1 || y_shape.ndim != 1 {
            return Err(GpuError::InvalidArgument);
        }
        let rows = w_shape.dims[0];
        let cols = w_shape.dims[1];
        if x_shape.dims[0] != cols || y_shape.dims[0] != rows {
            return Err(GpuError::InvalidBufferLength);
        }
        let x_i8 = x.as_i8_slice().ok_or(GpuError::InvalidBufferLength)?;
        let y_i32 = y.as_i32_slice_mut().ok_or(GpuError::InvalidBufferLength)?;
        self.submit_ternary_2bit_gemv(w.as_bytes(), &x_i8[..cols], &mut y_i32[..rows], rows, cols)
    }
}
