//! `#![no_std]` 64-byte aligned AMD CDNA3 Type-3 PM4 packet builder and compute ring.
//!
//! Provides deterministic packet encoding, count/header validation, ring wraparound,
//! release-fenced doorbell tail updates, and FNV-1a ring integrity digests with zero floats
//! and zero heap allocations.

use core::fmt;
use core::sync::atomic::{AtomicU64, Ordering, fence};

/// Packet Type-3 tag in bits `[31:30]` of the PM4 header DWORD.
pub const PM4_TYPE3_TAG: u32 = 3u32 << 30;
/// Compute shader-type flag (bit `[1]` of Type-3 PM4 header).
pub const PM4_SHADER_COMPUTE: u32 = 1u32 << 1;
/// Maximum Type-3 body DWORD count accepted by [`pm4_type3_header`] (`0x3FFF = 16_383`).
/// Reserving `0x3FFF` encoded count avoids ambiguity with hardware 1-DWORD NOP sentinels.
pub const PM4_MAX_BODY_WORDS: usize = 0x3FFF;

const FNV1A_OFFSET_BASIS: u64 = 0xCBF2_9CE4_8422_2325;
const FNV1A_PRIME: u64 = 0x0100_0000_01B3;

/// Standard AMD CDNA3 (MI300X) Type-3 PM4 opcodes.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Pm4Opcode {
    Nop = 0x10,
    DispatchDirect = 0x15,
    DispatchIndirect = 0x16,
    WriteData = 0x37,
    WaitRegMem = 0x3C,
    EventWrite = 0x46,
    ReleaseMem = 0x49,
    AcquireMem = 0x58,
    SetShReg = 0x76,
    SetUconfigReg = 0x79,
}

impl Pm4Opcode {
    /// Convert a raw 8-bit IT_OPCODE into a known [`Pm4Opcode`].
    #[inline]
    pub const fn from_u8(raw: u8) -> Option<Self> {
        match raw {
            0x10 => Some(Self::Nop),
            0x15 => Some(Self::DispatchDirect),
            0x16 => Some(Self::DispatchIndirect),
            0x37 => Some(Self::WriteData),
            0x3C => Some(Self::WaitRegMem),
            0x46 => Some(Self::EventWrite),
            0x49 => Some(Self::ReleaseMem),
            0x58 => Some(Self::AcquireMem),
            0x76 => Some(Self::SetShReg),
            0x79 => Some(Self::SetUconfigReg),
            _ => None,
        }
    }

    /// Raw 8-bit opcode value.
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Errors produced by PM4 packet construction or ring submission.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Pm4Error {
    /// Body word count is zero, exceeds [`PM4_MAX_BODY_WORDS`], mismatches the header, or exceeds ring capacity.
    InvalidCount,
    /// Packet header is not Type-3 (`[31:30] != 3`).
    InvalidHeader,
    /// Ring capacity is smaller than a minimum 2-DWORD packet.
    RingTooSmall,
    /// Ring does not currently have enough free slots for the packet.
    RingFull,
    /// Attempted to consume more words than currently queued in the ring.
    ConsumeUnderflow,
}

impl fmt::Display for Pm4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCount => f.write_str("invalid PM4 Type-3 packet word count"),
            Self::InvalidHeader => f.write_str("invalid PM4 packet header (expected Type-3)"),
            Self::RingTooSmall => f.write_str("PM4 ring capacity too small"),
            Self::RingFull => f.write_str("PM4 ring buffer full"),
            Self::ConsumeUnderflow => f.write_str("cannot consume more PM4 words than queued"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Pm4Error {}

/// Encode an AMD CDNA3 Type-3 PM4 header DWORD (`compute` shader flag set).
///
/// Returns `None` when `body_words == 0` or `body_words > PM4_MAX_BODY_WORDS`.
#[inline]
pub const fn pm4_type3_header(opcode: Pm4Opcode, body_words: usize) -> Option<u32> {
    if body_words == 0 || body_words > PM4_MAX_BODY_WORDS {
        return None;
    }
    let count_field = (body_words - 1) as u32;
    Some(
        PM4_TYPE3_TAG
            | ((count_field & 0x3FFF) << 16)
            | ((opcode.as_u8() as u32) << 8)
            | PM4_SHADER_COMPUTE,
    )
}

/// Decode `(opcode_byte, body_words)` from a Type-3 PM4 header DWORD.
#[inline]
pub const fn pm4_decode_type3_header(header: u32) -> Result<(u8, usize), Pm4Error> {
    if (header >> 30) != 3 {
        return Err(Pm4Error::InvalidHeader);
    }
    let count_field = ((header >> 16) & 0x3FFF) as usize;
    if count_field >= PM4_MAX_BODY_WORDS {
        return Err(Pm4Error::InvalidCount);
    }
    let opcode = ((header >> 8) & 0xFF) as u8;
    Ok((opcode, count_field + 1))
}

/// 64-byte aligned single cache-line AMD CDNA3 Type-3 PM4 packet frame (16 DWORDs).
#[repr(C, align(64))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Pm4Packet64 {
    pub words: [u32; 16],
}

impl Pm4Packet64 {
    /// Construct a `PKT3_NOP` packet with `body_words` zero payload DWORDs (`1..=15`).
    pub const fn nop(body_words: usize) -> Result<Self, Pm4Error> {
        if body_words == 0 || body_words > 15 {
            return Err(Pm4Error::InvalidCount);
        }
        let Some(hdr) = pm4_type3_header(Pm4Opcode::Nop, body_words) else {
            return Err(Pm4Error::InvalidCount);
        };
        let mut words = [0u32; 16];
        words[0] = hdr;
        Ok(Self { words })
    }

    /// Construct a `PKT3_DISPATCH_DIRECT` compute kernel launch packet.
    pub const fn dispatch_direct(
        dim_x: u32,
        dim_y: u32,
        dim_z: u32,
        dispatch_initiator: u32,
    ) -> Self {
        let hdr = match pm4_type3_header(Pm4Opcode::DispatchDirect, 4) {
            Some(h) => h,
            None => 0,
        };
        let mut words = [0u32; 16];
        words[0] = hdr;
        words[1] = dim_x;
        words[2] = dim_y;
        words[3] = dim_z;
        words[4] = dispatch_initiator;
        Self { words }
    }

    /// Construct a `PKT3_ACQUIRE_MEM` cache coherence / memory barrier packet.
    pub const fn acquire_mem(coher_cntl: u32, size_bytes: u64, base_addr: u64) -> Self {
        let hdr = match pm4_type3_header(Pm4Opcode::AcquireMem, 7) {
            Some(h) => h,
            None => 0,
        };
        let mut words = [0u32; 16];
        words[0] = hdr;
        words[1] = coher_cntl;
        words[2] = size_bytes as u32;
        words[3] = (size_bytes >> 32) as u32;
        words[4] = base_addr as u32;
        words[5] = (base_addr >> 32) as u32;
        words[6] = 0x0000_000A; // poll interval
        words[7] = 0; // GCR_CNTL
        Self { words }
    }

    /// Active DWORD slice for this packet (`1 + body_words`), validated from its header.
    pub fn active_slice(&self) -> Result<&[u32], Pm4Error> {
        let (_, body_words) = pm4_decode_type3_header(self.words[0])?;
        let total = 1 + body_words;
        if total > 16 {
            return Err(Pm4Error::InvalidCount);
        }
        Ok(&self.words[..total])
    }
}

/// 64-byte aligned fixed-capacity AMD CDNA3 Type-3 PM4 command ring buffer.
#[repr(C, align(64))]
pub struct Pm4Ring<const WORDS: usize> {
    words: [u32; WORDS],
    wptr: usize,
    rptr: usize,
    doorbell_tail: AtomicU64,
    packets_submitted: u64,
}

impl<const WORDS: usize> Default for Pm4Ring<WORDS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const WORDS: usize> Pm4Ring<WORDS> {
    /// Create an empty, zeroed, 64-byte aligned PM4 command ring.
    pub const fn new() -> Self {
        Self {
            words: [0u32; WORDS],
            wptr: 0,
            rptr: 0,
            doorbell_tail: AtomicU64::new(0),
            packets_submitted: 0,
        }
    }

    /// Ring storage capacity in 32-bit DWORDs.
    #[inline]
    pub const fn capacity_words(&self) -> usize {
        WORDS
    }

    /// Number of queued DWORDs awaiting consumption.
    #[inline]
    pub const fn used_words(&self) -> usize {
        self.wptr - self.rptr
    }

    /// Number of free DWORD slots remaining in the ring.
    #[inline]
    pub const fn free_words(&self) -> usize {
        WORDS.saturating_sub(self.wptr - self.rptr)
    }

    /// Monotonic write pointer in DWORDs.
    #[inline]
    pub const fn wptr(&self) -> usize {
        self.wptr
    }

    /// Monotonic read/consume pointer in DWORDs.
    #[inline]
    pub const fn rptr(&self) -> usize {
        self.rptr
    }

    /// Total number of valid Type-3 packets committed to the ring.
    #[inline]
    pub const fn packets_submitted(&self) -> u64 {
        self.packets_submitted
    }

    /// Latest release-fenced doorbell tail value (`wptr` after last packet commit).
    #[inline]
    pub fn doorbell_tail(&self) -> u64 {
        self.doorbell_tail.load(Ordering::Acquire)
    }

    /// Read a raw DWORD at physical ring slot `slot_idx % WORDS`.
    #[inline]
    pub fn peek_slot(&self, slot_idx: usize) -> u32 {
        if WORDS == 0 {
            return 0;
        }
        self.words[slot_idx % WORDS]
    }

    /// Submit a Type-3 PM4 packet from opcode and payload body DWORDs.
    ///
    /// Rejects `body.is_empty()`, `body.len() > PM4_MAX_BODY_WORDS`, or packets larger
    /// than `WORDS` with [`Pm4Error::InvalidCount`], handles ring wraparound, issues a
    /// `Release` memory fence, and updates the doorbell tail.
    pub fn submit_type3(&mut self, opcode: Pm4Opcode, body: &[u32]) -> Result<u64, Pm4Error> {
        if WORDS < 2 {
            return Err(Pm4Error::RingTooSmall);
        }
        let header = pm4_type3_header(opcode, body.len()).ok_or(Pm4Error::InvalidCount)?;
        let total_words = 1usize
            .checked_add(body.len())
            .ok_or(Pm4Error::InvalidCount)?;
        if total_words > WORDS {
            return Err(Pm4Error::InvalidCount);
        }
        if total_words > self.free_words() {
            return Err(Pm4Error::RingFull);
        }

        let start = self.wptr;
        self.words[start % WORDS] = header;
        let mut i = 0usize;
        while i < body.len() {
            self.words[(start + 1 + i) % WORDS] = body[i];
            i += 1;
        }

        fence(Ordering::Release);
        self.wptr = start + total_words;
        self.packets_submitted = self.packets_submitted.wrapping_add(1);
        let tail = self.wptr as u64;
        self.doorbell_tail.store(tail, Ordering::Release);
        Ok(tail)
    }

    /// Validate and submit a pre-encoded raw Type-3 packet slice `[header, body...]`.
    pub fn submit_raw_type3(&mut self, packet: &[u32]) -> Result<u64, Pm4Error> {
        if WORDS < 2 {
            return Err(Pm4Error::RingTooSmall);
        }
        if packet.is_empty() {
            return Err(Pm4Error::InvalidCount);
        }
        let (_, expected_body) = pm4_decode_type3_header(packet[0])?;
        let expected_total = 1usize
            .checked_add(expected_body)
            .ok_or(Pm4Error::InvalidCount)?;
        if packet.len() != expected_total || expected_total > WORDS {
            return Err(Pm4Error::InvalidCount);
        }
        if expected_total > self.free_words() {
            return Err(Pm4Error::RingFull);
        }

        let start = self.wptr;
        let mut i = 0usize;
        while i < packet.len() {
            self.words[(start + i) % WORDS] = packet[i];
            i += 1;
        }

        fence(Ordering::Release);
        self.wptr = start + expected_total;
        self.packets_submitted = self.packets_submitted.wrapping_add(1);
        let tail = self.wptr as u64;
        self.doorbell_tail.store(tail, Ordering::Release);
        Ok(tail)
    }

    /// Submit a 64-byte aligned [`Pm4Packet64`] using its header-encoded active length.
    pub fn submit_packet64(&mut self, packet: &Pm4Packet64) -> Result<u64, Pm4Error> {
        let slice = packet.active_slice()?;
        self.submit_raw_type3(slice)
    }

    /// Push a `PKT3_NOP` packet with `body_words` zero words.
    pub fn push_nop(&mut self, body_words: usize) -> Result<u64, Pm4Error> {
        if WORDS < 2 {
            return Err(Pm4Error::RingTooSmall);
        }
        let header = pm4_type3_header(Pm4Opcode::Nop, body_words).ok_or(Pm4Error::InvalidCount)?;
        let total_words = 1usize
            .checked_add(body_words)
            .ok_or(Pm4Error::InvalidCount)?;
        if total_words > WORDS {
            return Err(Pm4Error::InvalidCount);
        }
        if total_words > self.free_words() {
            return Err(Pm4Error::RingFull);
        }

        let start = self.wptr;
        self.words[start % WORDS] = header;
        let mut i = 0usize;
        while i < body_words {
            self.words[(start + 1 + i) % WORDS] = 0;
            i += 1;
        }

        fence(Ordering::Release);
        self.wptr = start + total_words;
        self.packets_submitted = self.packets_submitted.wrapping_add(1);
        let tail = self.wptr as u64;
        self.doorbell_tail.store(tail, Ordering::Release);
        Ok(tail)
    }

    /// Push a `PKT3_DISPATCH_DIRECT` compute grid launch packet.
    pub fn push_dispatch_direct(
        &mut self,
        dim_x: u32,
        dim_y: u32,
        dim_z: u32,
        dispatch_initiator: u32,
    ) -> Result<u64, Pm4Error> {
        self.submit_type3(
            Pm4Opcode::DispatchDirect,
            &[dim_x, dim_y, dim_z, dispatch_initiator],
        )
    }

    /// Push a `PKT3_ACQUIRE_MEM` synchronization barrier packet.
    pub fn push_acquire_mem(
        &mut self,
        coher_cntl: u32,
        size_bytes: u64,
        base_addr: u64,
    ) -> Result<u64, Pm4Error> {
        let pkt = Pm4Packet64::acquire_mem(coher_cntl, size_bytes, base_addr);
        self.submit_packet64(&pkt)
    }

    /// Push a `PKT3_WRITE_DATA` packet writing `data` DWORDs to `dst_addr`.
    pub fn push_write_data(&mut self, dst_addr: u64, data: &[u32]) -> Result<u64, Pm4Error> {
        if WORDS < 2 {
            return Err(Pm4Error::RingTooSmall);
        }
        if data.is_empty() {
            return Err(Pm4Error::InvalidCount);
        }
        let body_words = data
            .len()
            .checked_add(3)
            .ok_or(Pm4Error::InvalidCount)?;
        let header =
            pm4_type3_header(Pm4Opcode::WriteData, body_words).ok_or(Pm4Error::InvalidCount)?;
        let total_words = 1usize
            .checked_add(body_words)
            .ok_or(Pm4Error::InvalidCount)?;
        if total_words > WORDS {
            return Err(Pm4Error::InvalidCount);
        }
        if total_words > self.free_words() {
            return Err(Pm4Error::RingFull);
        }

        let start = self.wptr;
        self.words[start % WORDS] = header;
        self.words[(start + 1) % WORDS] = 0x0000_0500; // memory dest, increment address
        self.words[(start + 2) % WORDS] = dst_addr as u32;
        self.words[(start + 3) % WORDS] = (dst_addr >> 32) as u32;
        let mut i = 0usize;
        while i < data.len() {
            self.words[(start + 4 + i) % WORDS] = data[i];
            i += 1;
        }

        fence(Ordering::Release);
        self.wptr = start + total_words;
        self.packets_submitted = self.packets_submitted.wrapping_add(1);
        let tail = self.wptr as u64;
        self.doorbell_tail.store(tail, Ordering::Release);
        Ok(tail)
    }

    /// Advance the consumer read pointer by `word_count` DWORDs (simulating GPU ring drain).
    pub fn consume(&mut self, word_count: usize) -> Result<(), Pm4Error> {
        if word_count > self.used_words() {
            return Err(Pm4Error::ConsumeUnderflow);
        }
        self.rptr += word_count;
        Ok(())
    }

    /// Reset ring pointers, counters, and buffer contents.
    pub fn reset(&mut self) {
        self.words.fill(0);
        self.wptr = 0;
        self.rptr = 0;
        self.packets_submitted = 0;
        self.doorbell_tail.store(0, Ordering::Release);
    }

    /// Compute 64-bit FNV-1a digest over the physical ring buffer words and doorbell tail.
    pub fn fnv1a_ring_digest(&self) -> u64 {
        let mut h = FNV1A_OFFSET_BASIS;
        let mut i = 0usize;
        while i < WORDS {
            h = fnv1a_mix_u32(h, self.words[i]);
            i += 1;
        }
        h = fnv1a_mix_u64(h, self.doorbell_tail());
        fnv1a_mix_u64(h, self.wptr as u64)
    }

    /// Compute 64-bit FNV-1a digest over the logical queued words `[rptr..wptr]` in FIFO order.
    pub fn fnv1a_active_digest(&self) -> u64 {
        let mut h = FNV1A_OFFSET_BASIS;
        if WORDS == 0 {
            return h;
        }
        let mut idx = self.rptr;
        while idx < self.wptr {
            h = fnv1a_mix_u32(h, self.words[idx % WORDS]);
            idx += 1;
        }
        h
    }
}

#[inline]
const fn fnv1a_mix_u32(mut h: u64, val: u32) -> u64 {
    let bytes = val.to_le_bytes();
    let mut i = 0usize;
    while i < 4 {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(FNV1A_PRIME);
        i += 1;
    }
    h
}

#[inline]
const fn fnv1a_mix_u64(mut h: u64, val: u64) -> u64 {
    let bytes = val.to_le_bytes();
    let mut i = 0usize;
    while i < 8 {
        h ^= bytes[i] as u64;
        h = h.wrapping_mul(FNV1A_PRIME);
        i += 1;
    }
    h
}
