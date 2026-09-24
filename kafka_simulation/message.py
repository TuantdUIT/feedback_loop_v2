"""Schema message tự chứa input, output và nguồn capture trước khi đưa vào hàng đợi."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


LEVELS = tuple(f"L{i}" for i in range(1, 8))


def validate(msg: Any) -> list[str]:
    """Trả danh sách lỗi; danh sách rỗng nghĩa là message hợp lệ."""
    if not isinstance(msg, dict):
        return ["message phải là object"]
    errors = []
    if not isinstance(msg.get("msg_id"), str) or re.fullmatch(r".+#\d{5}", msg["msg_id"]) is None:
        errors.append("msg_id phải có dạng stem#NNNNN")
    if not isinstance(msg.get("text"), str) or not msg["text"]:
        errors.append("text phải là chuỗi không rỗng")
    result = msg.get("result")
    if not isinstance(result, dict):
        errors.append("result phải là object")
    else:
        for level in LEVELS:
            entries = result.get(level)
            if not isinstance(entries, list) or any(not isinstance(item, str) for item in entries):
                errors.append(f"result.{level} phải là list chuỗi")
    source = msg.get("source")
    if not isinstance(source, dict):
        errors.append("source phải là object")
    else:
        if not isinstance(source.get("capture"), str) or not source["capture"]:
            errors.append("source.capture phải là chuỗi")
        if not isinstance(source.get("endpoint"), str) or not source["endpoint"]:
            errors.append("source.endpoint phải là chuỗi")
        index = source.get("batch_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            errors.append("source.batch_index phải là số nguyên không âm")
    timestamp = msg.get("produced_at")
    if not isinstance(timestamp, str):
        errors.append("produced_at phải là thời gian ISO 8601")
    else:
        try:
            if datetime.fromisoformat(timestamp).tzinfo is None:
                errors.append("produced_at phải có múi giờ")
        except ValueError:
            errors.append("produced_at phải là thời gian ISO 8601")
    return errors
