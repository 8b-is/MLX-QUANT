import os
import tempfile

import pytest
from mlx.quant.shm_ring import SharedMemoryRingBuffer


def test_ring_create_and_push_pop():
    with tempfile.TemporaryDirectory() as tmpdir:
        ring_path = os.path.join(tmpdir, "test_ring.shm")
        ring = SharedMemoryRingBuffer(ring_path, capacity=16, create=True)

        # Push 5 records
        for i in range(5):
            ok = ring.push(
                event_type=1,
                node_id=100 + i,
                degree=float(i * 2.0),
                traversal_count=float(i * 10.0),
                payload_trits=f"trit_payload_{i}".encode("utf-8"),
            )
            assert ok is True

        head, tail = ring.get_pointers()
        assert head == 5
        assert tail == 0

        # Pop 3 records
        batch1 = ring.pop_batch(max_records=3)
        assert len(batch1) == 3
        assert batch1[0].node_id == 100
        assert batch1[1].node_id == 101
        assert batch1[2].node_id == 102

        head, tail = ring.get_pointers()
        assert head == 5
        assert tail == 3

        # Pop remaining
        batch2 = ring.pop_batch(max_records=10)
        assert len(batch2) == 2
        assert batch2[0].node_id == 103
        assert batch2[1].node_id == 104

        head, tail = ring.get_pointers()
        assert head == 5
        assert tail == 5

        # Empty pop
        assert ring.pop_batch() == []

        ring.close()


def test_ring_capacity_and_backpressure():
    with tempfile.TemporaryDirectory() as tmpdir:
        ring_path = os.path.join(tmpdir, "test_ring_bp.shm")
        ring = SharedMemoryRingBuffer(ring_path, capacity=4, create=True)

        # Fill ring to capacity
        for i in range(4):
            assert (
                ring.push(event_type=0, node_id=i, degree=1.0, traversal_count=1.0)
                is True
            )

        # 5th push must fail (backpressure)
        assert (
            ring.push(event_type=0, node_id=999, degree=1.0, traversal_count=1.0)
            is False
        )

        # Pop 2 records to relieve backpressure
        popped = ring.pop_batch(max_records=2)
        assert len(popped) == 2

        # Now push should succeed
        assert (
            ring.push(event_type=0, node_id=1000, degree=1.0, traversal_count=1.0)
            is True
        )

        ring.close()


def test_ring_producer_consumer_reopen():
    with tempfile.TemporaryDirectory() as tmpdir:
        ring_path = os.path.join(tmpdir, "test_ring_ipc.shm")
        producer = SharedMemoryRingBuffer(ring_path, capacity=32, create=True)
        consumer = SharedMemoryRingBuffer(ring_path, capacity=32, create=False)

        producer.push(event_type=2, node_id=42, degree=3.14, traversal_count=2.71)

        records = consumer.pop_batch(max_records=1)
        assert len(records) == 1
        assert records[0].node_id == 42
        assert abs(records[0].degree - 3.14) < 1e-4

        producer.close()
        consumer.close()
