//! Bounded 8-arm spiral command queue backed by [`BumpAllocator`] scratch storage and [`AgxDoorbell`].
//!
//! Replaces the RC1 `Vec`/`String` stubs with fixed stack-array `[Option<QueueArm<'a>>; 8]`,
//! fixed `[u8; 32]` arm names, allocator-backed descriptor scratch, and live GPU/simulator dispatch.

use crate::ExecutionMode;
use crate::agx_doorbell::AgxDoorbell;
use crate::allocator::BumpAllocator;
use crate::tensor::Tensor;
use crate::{copy_fixed_label, fnv1a_64};
use core::sync::atomic::{Ordering, fence};
use mlx_quant_gpu::GpuError;

/// Maximum number of spiral queue arms supported per [`PolarQueue`].
pub const MAX_POLAR_ARMS: usize = 8;
/// Maximum byte length of an arm's fixed ASCII/UTF-8 label.
pub const ARM_NAME_CAP: usize = 32;
const ARM_SCRATCH_BYTES: usize = 256;
const QUEUE_STATE_BYTES: usize = 512;

/// A single bounded stream arm inside [`PolarQueue`].
#[derive(Debug)]
pub struct QueueArm<'a> {
    name: [u8; ARM_NAME_CAP],
    name_len: usize,
    active_tensor: Option<Tensor<'a>>,
    ring_scratch: &'a mut [u8],
    dispatch_count: u64,
    last_digest: u64,
}

impl<'a> QueueArm<'a> {
    /// Arm identifier string.
    #[inline]
    pub fn name(&self) -> &str {
        core::str::from_utf8(&self.name[..self.name_len]).unwrap_or("arm")
    }

    /// Attached tensor, if any.
    #[inline]
    pub fn tensor(&self) -> Option<&Tensor<'a>> {
        self.active_tensor.as_ref()
    }

    /// Mutable access to the attached tensor, if any.
    #[inline]
    pub fn tensor_mut(&mut self) -> Option<&mut Tensor<'a>> {
        self.active_tensor.as_mut()
    }

    /// Allocator-carved 64-byte aligned ring scratch buffer for this arm.
    #[inline]
    pub fn scratch(&self) -> &[u8] {
        self.ring_scratch
    }

    /// Number of collapses/dispatches recorded on this arm.
    #[inline]
    pub const fn dispatch_count(&self) -> u64 {
        self.dispatch_count
    }

    /// Latest FNV-1a digest observed for this arm's attached tensor.
    #[inline]
    pub const fn last_digest(&self) -> u64 {
        self.last_digest
    }
}

/// Receipt emitted by [`PolarQueue::collapse_and_execute`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PolarExecutionReceipt {
    pub ring_id: u32,
    pub active_arms: usize,
    pub total_bytes: usize,
    pub combined_digest: u64,
    pub doorbell_count: u64,
    pub backend_mode: &'static str,
    pub mode: ExecutionMode,
}

/// Fixed-slot 8-arm spiral dispatch queue coordinating [`Tensor`] streams over [`AgxDoorbell`].
pub struct PolarQueue<'a> {
    arms: [Option<QueueArm<'a>>; MAX_POLAR_ARMS],
    arm_count: usize,
    ring_state: &'a mut [u8],
    allocator: &'a BumpAllocator,
    doorbell: AgxDoorbell,
}

impl<'a> PolarQueue<'a> {
    /// Construct a [`PolarQueue`], allocating ring descriptor state from `allocator`.
    /// Returns `None` if `allocator` does not have `512` bytes remaining.
    pub fn try_new(allocator: &'a BumpAllocator) -> Option<Self> {
        Self::with_doorbell(allocator, AgxDoorbell::new())
    }

    /// Construct a [`PolarQueue`] with an explicit [`AgxDoorbell`] (e.g., live Metal or simulator).
    pub fn with_doorbell(allocator: &'a BumpAllocator, doorbell: AgxDoorbell) -> Option<Self> {
        let ring_state = allocator.alloc_bytes(QUEUE_STATE_BYTES, 64)?;
        ring_state.fill(0);
        Some(Self {
            arms: [const { None }; MAX_POLAR_ARMS],
            arm_count: 0,
            ring_state,
            allocator,
            doorbell,
        })
    }

    /// Construct a [`PolarQueue`], panicking if `allocator` cannot provide `512` bytes of ring state.
    pub fn new(allocator: &'a BumpAllocator) -> Self {
        Self::try_new(allocator).expect("PolarQueue scratch allocation failed")
    }

    /// Attach a named spiral arm (max 8), allocating `256` bytes of 64-byte aligned scratch from `allocator`.
    pub fn attach_arm(&mut self, name: &str) -> Option<usize> {
        if self.arm_count >= MAX_POLAR_ARMS {
            return None;
        }
        let scratch = self.allocator.alloc_bytes(ARM_SCRATCH_BYTES, 64)?;
        scratch.fill(0);
        let (name_buf, name_len) = copy_fixed_label::<ARM_NAME_CAP>(name);
        let idx = self.arm_count;
        self.arms[idx] = Some(QueueArm {
            name: name_buf,
            name_len,
            active_tensor: None,
            ring_scratch: scratch,
            dispatch_count: 0,
            last_digest: 0,
        });
        self.arm_count += 1;
        Some(idx)
    }

    /// Feed a [`Tensor`] into spiral arm `arm_index`, returning `Err(tensor)` if out of bounds.
    pub fn feed_tensor(&mut self, arm_index: usize, tensor: Tensor<'a>) -> Result<(), Tensor<'a>> {
        let Some(Some(arm)) = self.arms.get_mut(arm_index) else {
            return Err(tensor);
        };
        let digest = tensor.fnv1a_digest();
        arm.last_digest = digest;
        if arm.ring_scratch.len() >= 16 {
            arm.ring_scratch[0..8].copy_from_slice(&digest.to_le_bytes());
            arm.ring_scratch[8..16].copy_from_slice(&(tensor.byte_len() as u64).to_le_bytes());
        }
        arm.active_tensor = Some(tensor);
        Ok(())
    }

    /// Immutable reference to spiral arm `arm_index`.
    #[inline]
    pub fn arm(&self, arm_index: usize) -> Option<&QueueArm<'a>> {
        self.arms.get(arm_index).and_then(|slot| slot.as_ref())
    }

    /// Number of currently attached spiral arms (`0..=8`).
    #[inline]
    pub const fn arm_count(&self) -> usize {
        self.arm_count
    }

    /// Allocator-carved queue ring descriptor bytes.
    #[inline]
    pub fn ring_state(&self) -> &[u8] {
        self.ring_state
    }

    /// Borrow the underlying [`AgxDoorbell`].
    #[inline]
    pub const fn doorbell(&self) -> &AgxDoorbell {
        &self.doorbell
    }

    /// Commit attached arm digests into allocator-backed ring state, issue a release fence,
    /// and ring the doorbell.
    pub fn collapse_and_execute(&mut self, ring_id: u32) -> PolarExecutionReceipt {
        let mut active_arms = 0usize;
        let mut total_bytes = 0usize;

        for (idx, slot) in self.arms.iter_mut().enumerate() {
            let Some(arm) = slot.as_mut() else {
                continue;
            };
            if let Some(tensor) = arm.active_tensor.as_ref() {
                let d = tensor.fnv1a_digest();
                let b = tensor.byte_len();
                arm.last_digest = d;
                arm.dispatch_count = arm.dispatch_count.wrapping_add(1);
                if arm.ring_scratch.len() >= 24 {
                    arm.ring_scratch[0..8].copy_from_slice(&d.to_le_bytes());
                    arm.ring_scratch[8..16].copy_from_slice(&(b as u64).to_le_bytes());
                    arm.ring_scratch[16..24].copy_from_slice(&arm.dispatch_count.to_le_bytes());
                }
                let state_off = idx * 16;
                if state_off + 16 <= self.ring_state.len() {
                    self.ring_state[state_off..state_off + 8].copy_from_slice(&d.to_le_bytes());
                    self.ring_state[state_off + 8..state_off + 16]
                        .copy_from_slice(&(b as u64).to_le_bytes());
                }
                active_arms += 1;
                total_bytes += b;
            }
        }

        fence(Ordering::Release);
        let doorbell_count = self.doorbell.ring(ring_id);
        let combined_digest = fnv1a_64(&self.ring_state[..MAX_POLAR_ARMS * 16]);
        PolarExecutionReceipt {
            ring_id,
            active_arms,
            total_bytes,
            combined_digest,
            doorbell_count,
            backend_mode: self.doorbell.backend_mode(),
            mode: self.doorbell.execution_mode(),
        }
    }

    /// Execute a real 2-bit ternary GEMV (`arms[weights_arm] * arms[act_arm] -> arms[out_arm]`)
    /// through [`AgxDoorbell`] (live Metal GPU on macOS, or portable ring simulator elsewhere).
    pub fn dispatch_ternary_gemv(
        &mut self,
        weights_arm: usize,
        act_arm: usize,
        out_arm: usize,
    ) -> Result<u64, GpuError> {
        if weights_arm >= MAX_POLAR_ARMS
            || act_arm >= MAX_POLAR_ARMS
            || out_arm >= MAX_POLAR_ARMS
            || weights_arm == out_arm
            || act_arm == out_arm
        {
            return Err(GpuError::InvalidArgument);
        }

        // Borrow disjoint arm slots safely by taking output tensor temporarily and restoring it.
        let mut out_tensor = self.arms[out_arm]
            .as_mut()
            .and_then(|arm| arm.active_tensor.take())
            .ok_or(GpuError::InvalidArgument)?;

        let result = {
            let w_tensor = self.arms[weights_arm]
                .as_ref()
                .and_then(|arm| arm.active_tensor.as_ref())
                .ok_or(GpuError::InvalidArgument);
            let x_tensor = self.arms[act_arm]
                .as_ref()
                .and_then(|arm| arm.active_tensor.as_ref())
                .ok_or(GpuError::InvalidArgument);
            match (w_tensor, x_tensor) {
                (Ok(w), Ok(x)) => self.doorbell.submit_tensor_gemv(w, x, &mut out_tensor),
                (Err(e), _) | (_, Err(e)) => Err(e),
            }
        };

        let out_digest = out_tensor.fnv1a_digest();
        if let Some(arm) = self.arms[out_arm].as_mut() {
            arm.last_digest = out_digest;
            arm.dispatch_count = arm.dispatch_count.wrapping_add(1);
            arm.active_tensor = Some(out_tensor);
        }

        if result.is_ok() {
            fence(Ordering::Release);
            if self.ring_state.len() >= (out_arm + 1) * 16 {
                let off = out_arm * 16;
                self.ring_state[off..off + 8].copy_from_slice(&out_digest.to_le_bytes());
            }
        }
        result
    }
}
