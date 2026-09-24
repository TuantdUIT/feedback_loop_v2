"""Kiểm tra hàng đợi FIFO và luồng producer -> queue -> consumer."""

import json

from kafka_simulation.consumer import run_consumer
from kafka_simulation.message_queue import MessageQueue
from kafka_simulation.producer import DEFAULT_TEMPLATE, run_producer


def test_fifo_order_and_empty_queue() -> None:
    queue = MessageQueue()
    for n in (1, 2, 3):
        queue.publish({"n": n})
    assert len(queue) == 3
    assert [queue.consume()["n"] for _ in range(3)] == [1, 2, 3]
    assert queue.consume() is None
    assert (queue.published, queue.consumed, len(queue)) == (3, 3, 0)


def test_template_flows_through_queue(tmp_path) -> None:
    queue = MessageQueue()
    stats = run_producer(queue, [DEFAULT_TEMPLATE])
    assert stats == {"sent": 46, "skipped": 0}
    assert len(queue) == 46

    output = tmp_path / "cases.json"
    records = run_consumer(queue, output)
    assert len(queue) == 0
    assert [r.case_id for r in records] == [f"template#{i:05d}" for i in range(46)]
    assert len(json.loads(output.read_text(encoding="utf-8"))) == 46
