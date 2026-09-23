"""Lấy hết message trong hàng đợi, chuyển sang CaseRecord và ghi ra file JSON."""

from __future__ import annotations

import json
from pathlib import Path

from feedback.core.schemas import CaseRecord
from kafka_simulation.bridge import to_case_record
from kafka_simulation.message_queue import MessageQueue


def run_consumer(queue: MessageQueue, output_path: str | Path) -> list[CaseRecord]:
    """Tiêu thụ tới khi hàng đợi rỗng rồi ghi toàn bộ CaseRecord một lần."""
    records: list[CaseRecord] = []
    while (msg := queue.consume()) is not None:
        records.append(to_case_record(msg))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([record.to_dict() for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records
