"""Giai đoạn ``compare``: so span model với span DeepSeek; giống hệt thì không cần judge.

``truncated`` bị bỏ qua khi so vì model đang triển khai không có tín hiệu này. Giống nhau KHÔNG có
nghĩa là đúng — kênh random audit lấy mẫu cả nhánh ``agreed``.
"""

from __future__ import annotations

from feedback.core.config import load_yaml
from feedback.core.diff_metrics import compare as diff_compare
from feedback.core.schemas import CaseRecord


def run(record: CaseRecord) -> tuple[CaseRecord, str]:
    weights = load_yaml("labels")["severity_weight"]
    record.diff = diff_compare(record.old.spans, record.new.spans, weights, ignore_truncated=True)
    return record, "agreed" if record.diff["identical"] else "to_judge"
