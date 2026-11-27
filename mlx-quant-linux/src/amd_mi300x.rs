//! Safe 64-byte aligned AMD CDNA3 (MI300X) Type-3 PM4 command ring wrapper.
//!
//! Uses [`mlx_quant_gpu::pm4::Pm4Ring`] for deterministic packet validation, wraparound,
//! release-fenced doorbell tail tracking, and FNV-1a ring digests. The in-memory PM4 ring
//! encoding is `Measured`; physical PCIe BAR2 MMIO doorbell submission is `DesignedDescriptor`.

use crate::ExecutionMode;
pub use mlx_quant_gpu::pm4::{Pm4Error, Pm4Opcode, Pm4Packet64, Pm4Ring};

/// Safe wrapper around a 64-byte aligned 1024-DWORD AMD CDNA3 Type-3 PM4 ring buffer.
#[repr(C, align(64))]
pub struct AmdComputeRing {
    ring: Pm4Ring<1024>,
    mmio_mode: ExecutionMode,
}

impl Default for AmdComputeRing {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl AmdComputeRing {
    /// Construct a fresh, zeroed, 64-byte aligned PM4 ring buffer.
    #[inline]
    pub const fn new() -> Self {
        Self {
            ring: Pm4Ring::new(),
            mmio_mode: ExecutionMode::DesignedDescriptor,
        }
    }

    /// Honest audit classification for physical MI300X PCIe MMIO transport.
    #[inline]
    pub const fn mmio_execution_mode(&self) -> ExecutionMode {
        self.mmio_mode
    }

    /// Enqueue a `PKT3_DISPATCH_DIRECT` compute grid launch packet, returning updated doorbell tail.
    #[inline]
    pub fn dispatch_kernel(
        &mut self,
        dim_x: u32,
        dim_y: u32,
        dim_z: u32,
        dispatch_initiator: u32,
    ) -> Result<u64, Pm4Error> {
        self.ring
            .push_dispatch_direct(dim_x, dim_y, dim_z, dispatch_initiator)
    }

    /// Enqueue an explicit 64-byte aligned [`Pm4Packet64`] frame.
    #[inline]
    pub fn dispatch_packet(&mut self, packet: &Pm4Packet64) -> Result<u64, Pm4Error> {
        self.ring.submit_packet64(packet)
    }

    /// Enqueue a `PKT3_ACQUIRE_MEM` cache coherence / memory barrier packet.
    #[inline]
    pub fn acquire_barrier(
        &mut self,
        coher_cntl: u32,
        size_bytes: u64,
        base_addr: u64,
    ) -> Result<u64, Pm4Error> {
        self.ring.push_acquire_mem(coher_cntl, size_bytes, base_addr)
    }

    /// Enqueue a `PKT3_WRITE_DATA` memory write packet.
    #[inline]
    pub fn write_data(&mut self, dst_addr: u64, data: &[u32]) -> Result<u64, Pm4Error> {
        self.ring.push_write_data(dst_addr, data)
    }

    /// Enqueue a `PKT3_NOP` packet with `body_words` zero words.
    #[inline]
    pub fn push_nop(&mut self, body_words: usize) -> Result<u64, Pm4Error> {
        self.ring.push_nop(body_words)
    }

    /// Drain all currently queued DWORDs, advancing `rptr` to `wptr` and returning words drained.
    #[inline]
    pub fn drain(&mut self) -> usize {
        let used = self.ring.used_words();
        if used > 0 {
            let _ = self.ring.consume(used);
        }
        used
    }

    /// Consume an exact number of queued DWORDs.
    #[inline]
    pub fn consume(&mut self, word_count: usize) -> Result<(), Pm4Error> {
        self.ring.consume(word_count)
    }

    /// Latest release-fenced doorbell tail (`wptr`).
    #[inline]
    pub fn doorbell_tail(&self) -> u64 {
        self.ring.doorbell_tail()
    }

    /// Deterministic FNV-1a digest over the queued FIFO words `[rptr..wptr]`.
    #[inline]
    pub fn fnv1a_active_digest(&self) -> u64 {
        self.ring.fnv1a_active_digest()
    }

    /// Deterministic FNV-1a digest over the full physical ring array and doorbell tail.
    #[inline]
    pub fn fnv1a_ring_digest(&self) -> u64 {
        self.ring.fnv1a_ring_digest()
    }

    /// Number of queued DWORDs awaiting drain.
    #[inline]
    pub const fn used_words(&self) -> usize {
        self.ring.used_words()
    }

    /// Number of free DWORD slots remaining.
    #[inline]
    pub const fn free_words(&self) -> usize {
        self.ring.free_words()
    }

    /// Cumulative packets submitted into the ring.
    #[inline]
    pub const fn packets_submitted(&self) -> u64 {
        self.ring.packets_submitted()
    }

    /// Read-only access to the underlying `Pm4Ring<1024>`.
    #[inline]
    pub const fn inner(&self) -> &Pm4Ring<1024> {
        &self.ring
    }
}
