//! Fixed-buffer stigmergic swarm trace (`DesignedDescriptor` for distributed RDMA/DHT gossip).

use crate::{ExecutionMode, copy_fixed_label};
use mlx_quant_core::ratio::splitmix64;

/// Fixed-size stigmergic pheromone state without heap `String`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PheromoneTrace {
    pub swarm_id: [u8; 32],
    pub swarm_id_len: usize,
    pub layer_idx: usize,
    pub state_digest: u64,
    pub deposits: u64,
    pub mode: ExecutionMode,
}

impl PheromoneTrace {
    /// Swarm identifier string slice.
    #[inline]
    pub fn swarm_name(&self) -> &str {
        core::str::from_utf8(&self.swarm_id[..self.swarm_id_len]).unwrap_or("swarm")
    }
}

/// Stigmergic distributed layer-completion tracer.
pub struct Multiverse;

impl Multiverse {
    /// Initialize a local stigmergic trace for `swarm_id` (network DHT/RDMA transport is `DesignedDescriptor`).
    pub fn join_swarm(swarm_id: &str, layer_idx: usize, initial_digest: u64) -> PheromoneTrace {
        let (buf, swarm_id_len) = copy_fixed_label::<32>(swarm_id);
        let mut h = initial_digest;
        for &b in &buf[..swarm_id_len] {
            h = splitmix64(h ^ u64::from(b));
        }
        PheromoneTrace {
            swarm_id: buf,
            swarm_id_len,
            layer_idx,
            state_digest: h,
            deposits: 0,
            mode: ExecutionMode::DesignedDescriptor,
        }
    }

    /// Record completion of `layer_idx` into the local pheromone state digest using SplitMix64.
    pub fn deposit_pheromone(trace: &mut PheromoneTrace, layer_idx: usize, layer_digest: u64) {
        trace.layer_idx = layer_idx;
        trace.state_digest = splitmix64(trace.state_digest ^ layer_digest ^ (layer_idx as u64));
        trace.deposits = trace.deposits.wrapping_add(1);
    }
}
