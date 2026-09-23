"""Đọc nhiều capture, kiểm tra message và đưa từng địa chỉ vào hàng đợi."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from feedback.core.normalize import make_case_id
from kafka_simulation.capture_parser import parse_capture
from kafka_simulation.message import validate
from kafka_simulation.message_queue import MessageQueue


DEFAULT_TEMPLATE = Path(__file__).with_name("template.txt")


def run_producer(
    queue: MessageQueue,
    capture_paths: list[str | Path] | tuple[str | Path, ...] = (DEFAULT_TEMPLATE,),
) -> dict[str, int]:
    """Ghép input/output ở parser, bỏ message hỏng và đếm số đã gửi."""
    sent = skipped = 0
    for path in capture_paths:
        source = Path(path)
        for payload in parse_capture(source):
            index = payload["source"]["batch_index"]
            msg = {
                "msg_id": make_case_id(source.name, index),
                "text": payload["text"],
                "result": payload["result"],
                "confidence": payload["confidence"],
                "source": payload["source"],
                "produced_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            errors = validate(msg)
            if errors:
                skipped += 1
                log_line = f"Bỏ {msg['msg_id']}: {', '.join(errors)}"
                print(log_line.encode("ascii", "backslashreplace").decode("ascii"), file=sys.stderr)
                continue
            queue.publish(msg)
            sent += 1
    return {"sent": sent, "skipped": skipped}
