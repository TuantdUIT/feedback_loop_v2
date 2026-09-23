"""Entry point: capture -> producer -> hàng đợi -> consumer -> CaseRecord, chưa gọi cascade."""

from __future__ import annotations

import argparse
from pathlib import Path

from kafka_simulation.consumer import run_consumer
from kafka_simulation.message_queue import MessageQueue
from kafka_simulation.producer import DEFAULT_TEMPLATE, run_producer


def main() -> None:
    """Chạy producer rồi consumer trên cùng một hàng đợi trong bộ nhớ."""
    parser = argparse.ArgumentParser(description="Luân chuyển output NER qua message queue mô phỏng")
    parser.add_argument("captures", nargs="*", type=Path, help="Đường dẫn capture curl")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("out") / "cases.json")
    args = parser.parse_args()

    queue = MessageQueue("ner.raw")
    stats = run_producer(queue, args.captures or [DEFAULT_TEMPLATE])
    records = run_consumer(queue, args.output)

    spans = sum(len(record.old.spans) for record in records)
    unlocated = sum(len(record.meta.get("unlocated", [])) for record in records)
    print(f"Queue '{queue.name}': publish {queue.published} | consume {queue.consumed} | còn lại {len(queue)}")
    print(f"{len(records)} CaseRecord, {spans} span, {unlocated} unlocated | bỏ qua {stats['skipped']} message hỏng")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
