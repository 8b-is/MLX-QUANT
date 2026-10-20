# Copyright © 2026 Peter Lodri / MLX-QUANT Contributors.
"""Offline ring regressions: no MLX native runtime, GPU or model is needed."""

import importlib.util
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "python/mlx/quant/shm_ring.py"
SPEC = importlib.util.spec_from_file_location("shm_ring_under_test", SOURCE)
ring_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ring_module
SPEC.loader.exec_module(ring_module)
Ring = ring_module.SharedMemoryRingBuffer


class SharedMemoryAttachTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "ring.shm"

    def ring(self, **kwargs):
        ring = Ring(str(self.path), **kwargs)
        self.addCleanup(ring.close)
        return ring

    def assert_fd_closed(self, fd):
        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_default_attach_small_ring(self):
        producer = self.ring(capacity=8, create=True)
        consumer = self.ring()
        self.assertEqual(consumer.capacity, 8)
        self.assertEqual(consumer.total_size, 64 + 8 * 64)
        self.assertTrue(producer.push(2, 42, 3.0, 4.0))
        self.assertEqual(consumer.pop_batch()[0].node_id, 42)

    def test_default_attach_large_ring_and_wrap(self):
        producer = self.ring(capacity=2048, create=True)
        consumer = self.ring()
        self.assertEqual(consumer.total_size, 64 + 2048 * 64)
        for i in range(2048):
            self.assertTrue(producer.push(1, i, 2.0, 3.0))
        self.assertFalse(producer.push(1, 9999, 2.0, 3.0))
        self.assertEqual(
            [r.node_id for r in consumer.pop_batch(2048)], list(range(2048))
        )
        for i in range(16):
            self.assertTrue(producer.push(1, i + 2048, 2.0, 3.0))
        self.assertEqual([r.seq for r in consumer.pop_batch()], list(range(2048, 2064)))
        self.assertEqual(consumer.pop_batch(), [])

    def test_stored_capacity_overrides_caller_hint(self):
        producer = self.ring(capacity=16, create=True)
        for capacity in [1, 32, 1024]:
            with self.subTest(capacity=capacity):
                consumer = self.ring(capacity=capacity)
                self.assertEqual(consumer.capacity, 16)
                self.assertTrue(producer.push(0, capacity, 0, 0))
                self.assertEqual(consumer.pop_batch()[0].node_id, capacity)
                consumer.close()
                self.assertTrue(self.path.exists())

    def test_invalid_creation_preserves_existing_file(self):
        self.path.write_bytes(b"keep this file")
        for capacity in [0, -1, 2**32, 1.5]:
            with self.subTest(capacity=capacity):
                with self.assertRaises(ValueError):
                    Ring(str(self.path), capacity=capacity, create=True)
                self.assertEqual(self.path.read_bytes(), b"keep this file")

    def test_invalid_files_rejected_before_mapping_without_fd_leak(self):
        def header(magic=ring_module.HEADER_MAGIC, capacity=8, size=64):
            return struct.pack(
                ring_module.HEADER_FMT, magic, capacity, size, 0, 0, 0, b""
            )

        files = {
            "short header": b"T3RING01",
            "bad magic": header(magic=b"INVALID!") + bytes(8 * 64),
            "record size": header(size=32) + bytes(8 * 64),
            "zero capacity": header(capacity=0),
            "truncated records": header() + bytes(8 * 64 - 1),
            "huge capacity": header(capacity=2**32 - 1),
        }
        real_open = os.open
        for reason, data in files.items():
            with self.subTest(reason=reason):
                self.path.write_bytes(data)
                descriptors = []

                def tracked_open(*args, **kwargs):
                    fd = real_open(*args, **kwargs)
                    descriptors.append(fd)
                    return fd

                with patch.object(ring_module.os, "open", tracked_open), patch.object(
                    ring_module.mmap, "mmap"
                ) as mapping:
                    with self.assertRaises(ValueError):
                        Ring(str(self.path))
                    mapping.assert_not_called()
                self.assertEqual(len(descriptors), 1)
                self.assert_fd_closed(descriptors[0])
                self.assertEqual(self.path.read_bytes(), data)

    def test_mapping_failure_closes_fd(self):
        self.ring(capacity=8, create=True)
        real_open = os.open
        descriptors = []

        def tracked_open(*args, **kwargs):
            fd = real_open(*args, **kwargs)
            descriptors.append(fd)
            return fd

        with patch.object(ring_module.os, "open", tracked_open), patch.object(
            ring_module.mmap, "mmap", side_effect=OSError("synthetic mmap failure")
        ):
            with self.assertRaisesRegex(OSError, "synthetic mmap failure"):
                Ring(str(self.path))
        self.assert_fd_closed(descriptors[0])

    def test_header_initialization_failure_closes_mapping_and_fd(self):
        real_map = ring_module.mmap.mmap
        resources = []

        def tracked_map(fd, *args, **kwargs):
            mapping = real_map(fd, *args, **kwargs)
            resources.append((fd, mapping))
            return mapping

        with patch.object(ring_module.mmap, "mmap", tracked_map), patch.object(
            Ring, "_init_header", side_effect=OSError("synthetic header failure")
        ):
            with self.assertRaisesRegex(OSError, "synthetic header failure"):
                Ring(str(self.path), capacity=8, create=True)
        self.assertTrue(resources[0][1].closed)
        self.assert_fd_closed(resources[0][0])

    def test_capacity_change_during_attach_closes_resources(self):
        self.ring(capacity=8, create=True)
        real_map = ring_module.mmap.mmap
        resources = []

        def changed_map(fd, *args, **kwargs):
            mapping = real_map(fd, *args, **kwargs)
            resources.append((fd, mapping))
            mapping[8:12] = struct.pack("=I", 4)
            return mapping

        with patch.object(ring_module.mmap, "mmap", changed_map):
            with self.assertRaisesRegex(ValueError, "capacity changed"):
                Ring(str(self.path))
        self.assertTrue(resources[0][1].closed)
        self.assert_fd_closed(resources[0][0])

    def test_close_is_repeatable_and_attach_preserves_file(self):
        self.ring(capacity=1024, create=True)
        consumer = self.ring()
        fd = consumer._fd
        consumer.close()
        consumer.close()
        self.assert_fd_closed(fd)
        self.assertTrue(self.path.exists())


if __name__ == "__main__":
    unittest.main()
