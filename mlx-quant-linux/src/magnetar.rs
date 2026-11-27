//! Honest userspace descriptors for memory page locking (`mlock`) and core affinity pinning.

use crate::ExecutionMode;
use crate::tensor::Tensor;

/// Descriptor recording a tensor memory-lock request (OS `mlock` requires privileged/platform hooks).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MemoryLockReceipt {
    pub addr: usize,
    pub byte_len: usize,
    pub locked: bool,
    pub mode: ExecutionMode,
}

/// Descriptor recording a thread/core affinity pinning request.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CoreAffinityReceipt {
    pub core_id: usize,
    pub pinned: bool,
    pub mode: ExecutionMode,
}

/// Page-locking and P-core affinity policy descriptors.
pub struct Magnetar;

impl Magnetar {
    /// Record an honest userspace memory-lock descriptor for `tensor` (`locked: false`, `DesignedDescriptor`).
    #[inline]
    pub fn lock_memory(tensor: &Tensor<'_>) -> MemoryLockReceipt {
        MemoryLockReceipt {
            addr: tensor.as_bytes().as_ptr() as usize,
            byte_len: tensor.byte_len(),
            locked: false,
            mode: ExecutionMode::DesignedDescriptor,
        }
    }

    /// Record an honest core-affinity pinning descriptor (`pinned: false`, `DesignedDescriptor`).
    #[inline]
    pub const fn pin_to_core(core_id: usize) -> CoreAffinityReceipt {
        CoreAffinityReceipt {
            core_id,
            pinned: false,
            mode: ExecutionMode::DesignedDescriptor,
        }
    }
}
