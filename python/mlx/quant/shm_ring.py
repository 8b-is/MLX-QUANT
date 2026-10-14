# Copyright © 2026 Peter Lodri / MLX-QUANT Contributors.
# Vaimshuk Step 2: Bounded Shared-Memory Ring Buffer (T3-SHM-RING01).

import mmap
import os
import struct
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Header Format:
# magic (8s) = b"T3RING01"
# capacity (I) = uint32
# record_size (I) = uint32
# head (Q) = uint64
# tail (Q) = uint64
# flags (Q) = uint64
# reserved (24s) = 24 bytes padding -> Total 64 bytes
HEADER_MAGIC = b"T3RING01"
HEADER_FMT = "=8sIIQQQ24s"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 64 bytes

# Record Format (64 bytes):
# seq (Q) = uint64
# timestamp_ns (Q) = uint64
# event_type (I) = uint32 (0=Edge, 1=DensityDelta, 2=GradientShock, 3=Probe)
# node_id (I) = uint32
# degree (f) = float32
# traversal_count (f) = float32
# payload_trits (16s) = 16 bytes
# extra (16s) = 16 bytes
RECORD_FMT = "=QQIIff16s16s"
RECORD_SIZE = struct.calcsize(RECORD_FMT)  # 64 bytes

assert HEADER_SIZE == 64, f"Header size must be 64 bytes, got {HEADER_SIZE}"
assert RECORD_SIZE == 64, f"Record size must be 64 bytes, got {RECORD_SIZE}"


@dataclass
class RingRecord:
    seq: int
    timestamp_ns: int
    event_type: int
    node_id: int
    degree: float
    traversal_count: float
    payload_trits: bytes = b"\x00" * 16
    extra: bytes = b"\x00" * 16


class SharedMemoryRingBuffer:
    """
    Bounded, lockless-ready memory-mapped ring buffer for low-latency Go <-> MLX IPC.
    Preserves authoritative Go t3: stream semantics while enabling zero-copy ingestion in MLX.
    """

    def __init__(self, filepath: str, capacity: int = 1024, create: bool = False):
        # Capacity describes a new ring only. Attach uses the on-disk geometry.
        if create and (not isinstance(capacity, int) or not 0 < capacity < 2**32):
            raise ValueError("Ring capacity must be a positive uint32")
        self.filepath = filepath
        self.capacity = capacity
        self.is_owner = create
        self._fd = None
        self._mmap = None

        try:
            flags = os.O_RDWR | (os.O_CREAT | os.O_TRUNC if create else 0)
            self._fd = os.open(filepath, flags, 0o666)
            if not create:
                self.capacity = self._header_capacity(
                    os.pread(self._fd, HEADER_SIZE, 0)
                )
            self.total_size = HEADER_SIZE + (self.capacity * RECORD_SIZE)
            if create:
                os.ftruncate(self._fd, self.total_size)
            elif os.fstat(self._fd).st_size < self.total_size:
                raise ValueError("Truncated ring file: capacity exceeds file size")
            self._mmap = mmap.mmap(
                self._fd,
                self.total_size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
            )
            if create:
                self._init_header()
            else:
                self._verify_header()
        except BaseException:
            self._close_resources()
            raise

    def _init_header(self):
        header_bytes = struct.pack(
            HEADER_FMT,
            HEADER_MAGIC,
            self.capacity,
            RECORD_SIZE,
            0,  # head
            0,  # tail
            0,  # flags
            b"\x00" * 24,
        )
        self._mmap[0:HEADER_SIZE] = header_bytes
        self._mmap.flush()

    @staticmethod
    def _header_capacity(header):
        if len(header) != HEADER_SIZE:
            raise ValueError("Truncated ring header")
        magic, cap, rsize, _, _, _, _ = struct.unpack(HEADER_FMT, header)
        if magic != HEADER_MAGIC:
            raise ValueError(f"Invalid ring magic: {magic}")
        if rsize != RECORD_SIZE:
            raise ValueError(
                f"Mismatched record size: expected {RECORD_SIZE}, got {rsize}"
            )
        if cap == 0:
            raise ValueError("Ring capacity must be positive")
        return cap

    def _verify_header(self):
        # A producer may advance its pointers, but cannot resize an attached ring.
        if self._header_capacity(self._mmap[0:HEADER_SIZE]) != self.capacity:
            raise ValueError("Ring capacity changed during attach")

    def get_pointers(self) -> Tuple[int, int]:
        """Reads head and tail atomically from header."""
        _, _, _, head, tail, _, _ = struct.unpack(HEADER_FMT, self._mmap[0:HEADER_SIZE])
        return head, tail

    def _set_head(self, head: int):
        self._mmap[16:24] = struct.pack("=Q", head)

    def _set_tail(self, tail: int):
        self._mmap[24:32] = struct.pack("=Q", tail)

    def push(
        self,
        event_type: int,
        node_id: int,
        degree: float,
        traversal_count: float,
        payload_trits: bytes = b"\x00" * 16,
    ) -> bool:
        """
        Pushes a new record to the ring buffer.
        Returns True if successful, False if the ring buffer is full (backpressure).
        """
        head, tail = self.get_pointers()
        if head - tail >= self.capacity:
            return False  # Bounded ring is full

        slot = head % self.capacity
        offset = HEADER_SIZE + (slot * RECORD_SIZE)

        rec_bytes = struct.pack(
            RECORD_FMT,
            head,
            time.time_ns(),
            event_type,
            node_id,
            float(degree),
            float(traversal_count),
            payload_trits[:16].ljust(16, b"\x00"),
            b"\x00" * 16,
        )
        self._mmap[offset : offset + RECORD_SIZE] = rec_bytes
        self._set_head(head + 1)
        return True

    def pop_batch(self, max_records: int = 128) -> List[RingRecord]:
        """
        Pops a batch of available records from the ring buffer.
        Advances tail pointer.
        """
        head, tail = self.get_pointers()
        available = head - tail
        if available <= 0:
            return []

        count = min(available, max_records)
        records = []

        for i in range(count):
            slot = (tail + i) % self.capacity
            offset = HEADER_SIZE + (slot * RECORD_SIZE)
            rec_data = self._mmap[offset : offset + RECORD_SIZE]
            seq, ts, ev_type, node_id, deg, trav, trits, extra = struct.unpack(
                RECORD_FMT, rec_data
            )
            records.append(
                RingRecord(
                    seq=seq,
                    timestamp_ns=ts,
                    event_type=ev_type,
                    node_id=node_id,
                    degree=deg,
                    traversal_count=trav,
                    payload_trits=trits,
                    extra=extra,
                )
            )

        self._set_tail(tail + count)
        return records

    def _close_resources(self):
        try:
            if self._mmap is not None:
                self._mmap.close()
        finally:
            self._mmap = None
            if self._fd is not None:
                fd, self._fd = self._fd, None
                os.close(fd)

    def close(self):
        try:
            self._close_resources()
        except Exception:
            pass
        if self.is_owner and os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except Exception:
                pass
