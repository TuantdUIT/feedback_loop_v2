"""Xuất case lỗi ĐÃ XÁC NHẬN thành ứng viên dữ liệu retrain (nhãn silver).

Chỉ lấy case ``deepseek_better`` có ``gate.confirmed_error == True`` — tức đã có ngưỡng severity hiệu
chuẩn. Trước khi hiệu chuẩn, không case nào đủ điều kiện và script xuất file rỗng: đó là hành vi đúng,
không phải lỗi. Nhãn đề xuất là bản DeepSeek được judge chọn, CHƯA qua người. Run ``calibration``
không bao giờ được xuất, để tập dùng đo judge không lọt vào dữ liệu train.

  .venv-layer1/Scripts/python.exe -m feedback.scripts.export_silver <run_id> [<run_id> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from feedback.core.config import PROJECT_ROOT
from feedback.core.schemas import Decision

SILVER = PROJECT_ROOT / "feedback" / "store" / "silver"


def silver_rows(run: dict[str, Any], records: list) -> list[dict[str, Any]]:
    if run["kind"] == "calibration":
        return []
    rows = []
    for record in records:
        gate = record.layers.get("gate", {})
        if record.decision is Decision.DEEPSEEK_BETTER and gate.get("confirmed_error") is True:
            rows.append({"case_id": record.case_id, "input": record.raw_text,
                         "tokens": record.new.tokens, "bio": record.new.bio,
                         "spans": [s.to_dict() for s in record.new.spans],
                         "label_source": "deepseek_judged_by_qwen", "human_verified": False,
                         "model_spans": [s.to_dict() for s in record.old.spans],
                         "severity": gate.get("severity_total")})
    return rows


def main() -> int:
    from feedback.pipeline.runner import Pipeline

    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+")
    args = parser.parse_args()
    pipeline = Pipeline()
    SILVER.mkdir(parents=True, exist_ok=True)
    for run_id in args.run_ids:
        run = pipeline.store.get_run(run_id)
        rows = silver_rows(run, pipeline.store.records(run_id))
        out = SILVER / f"{run_id}.json"
        out.write_text(json.dumps({"_meta": {**run["meta"], "run_id": run_id, "kind": run["kind"],
                                             "thresholds_version": run["meta"].get("thresholds_version")},
                                   "records": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        note = " (run hiệu chuẩn — không xuất)" if run["kind"] == "calibration" else ""
        print(f"{run_id}: {len(rows)} case{note} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
