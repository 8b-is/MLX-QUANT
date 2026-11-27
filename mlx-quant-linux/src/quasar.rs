//! RDMA outbound tensor transfer descriptor with live FNV-1a payload integrity digest.

use crate::tensor::Tensor;
use crate::{ExecutionMode, copy_fixed_label};

/// Outbound RDMA transfer receipt for a [`Tensor`] payload.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RdmaJetReceipt {
    pub byte_len: usize,
    pub payload_fnv1a: u64,
    pub target_buf: [u8; 32],
    pub target_len: usize,
    pub mode: ExecutionMode,
}

impl RdmaJetReceipt {
    /// Target endpoint string slice.
    #[inline]
    pub fn target_str(&self) -> &str {
        core::str::from_utf8(&self.target_buf[..self.target_len]).unwrap_or("invalid")
    }
}

/// Relativistic RDMA jet descriptor builder.
pub struct Quasar;

impl Quasar {
    /// Compute the exact payload FNV-1a digest for `tensor` and emit a `DesignedDescriptor`
    /// RDMA transmission receipt targeting `target_ip`.
    pub fn emit_rdma_jet(tensor: &Tensor<'_>, target_ip: &str) -> RdmaJetReceipt {
        let (target_buf, target_len) = copy_fixed_label::<32>(target_ip);
        RdmaJetReceipt {
            byte_len: tensor.byte_len(),
            payload_fnv1a: tensor.fnv1a_digest(),
            target_buf,
            target_len,
            mode: ExecutionMode::DesignedDescriptor,
        }
    }
}
