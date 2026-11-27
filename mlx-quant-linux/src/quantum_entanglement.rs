//! RDMA tensor memory registration descriptor (`DesignedDescriptor` without hardware NIC verbs).

use crate::tensor::Tensor;
use crate::{ExecutionMode, copy_fixed_label};
use mlx_quant_core::ratio::splitmix64;

/// Fixed-size RDMA region descriptor binding a local tensor digest to a remote IP/endpoint label.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RdmaRegionDescriptor {
    pub local_addr: usize,
    pub byte_len: usize,
    pub remote_endpoint: [u8; 32],
    pub endpoint_len: usize,
    pub rkey_hash: u64,
    pub mode: ExecutionMode,
}

impl RdmaRegionDescriptor {
    /// Remote endpoint string slice.
    #[inline]
    pub fn remote_str(&self) -> &str {
        core::str::from_utf8(&self.remote_endpoint[..self.endpoint_len]).unwrap_or("invalid")
    }
}

/// RDMA entanglement descriptor generator.
pub struct QuantumEntanglement;

impl QuantumEntanglement {
    /// Build a deterministic RDMA registration descriptor for `tensor` targeting `remote_ip`.
    pub fn entangle_tensors(tensor: &Tensor<'_>, remote_ip: &str) -> RdmaRegionDescriptor {
        let (remote_endpoint, endpoint_len) = copy_fixed_label::<32>(remote_ip);
        let mut h = tensor.fnv1a_digest();
        for &b in remote_ip.as_bytes() {
            h = splitmix64(h ^ u64::from(b));
        }
        RdmaRegionDescriptor {
            local_addr: tensor.as_bytes().as_ptr() as usize,
            byte_len: tensor.byte_len(),
            remote_endpoint,
            endpoint_len,
            rkey_hash: h,
            mode: ExecutionMode::DesignedDescriptor,
        }
    }
}
