"""Giai đoạn ``gate``: quyết định cuối cho một case (BUILD_PIPELINE.md §7.1).

``deepseek_better`` chỉ thành LỖI ĐÃ XÁC NHẬN của model khi severity ≥ ``thresholds.gate.min_severity``.
Ngưỡng đó phải đến từ hiệu chuẩn; khi còn ``null`` thì ``confirmed_error`` là ``None`` — nghĩa là
"ứng viên, chưa hiệu chuẩn", không phải "không lỗi".
"""

from __future__ import annotations

from typing import Any

from feedback.core.schemas import CaseRecord, Decision

VERDICT_TO_DECISION = {"model": Decision.MODEL_BETTER, "deepseek": Decision.DEEPSEEK_BETTER, "tie": Decision.TIE}


def run(record: CaseRecord, thresholds: dict[str, Any]) -> tuple[CaseRecord, str]:
    if record.diff.get("identical"):
        record.decision = Decision.AGREE
    else:
        verdict = record.layers.get("judge", {}).get("verdict")
        record.decision = VERDICT_TO_DECISION.get(verdict, Decision.INCONCLUSIVE)

    min_severity = (thresholds.get("gate") or {}).get("min_severity")
    severity = record.diff.get("severity_total", 0.0)
    confirmed: bool | None = False
    if record.decision is Decision.DEEPSEEK_BETTER:
        confirmed = None if min_severity is None else severity >= min_severity
    record.layers["gate"] = {"decision": record.decision.value, "severity_total": severity,
                             "min_severity": min_severity, "confirmed_error": confirmed,
                             "calibrated": min_severity is not None}
    return record, "decided"
