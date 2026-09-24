"""Quét ngưỡng severity trên một lượt hiệu chuẩn và đề xuất ``gate.min_severity``.

Với mỗi ngưỡng t: "lỗi đã xác nhận" = ``deepseek_better`` ∧ severity ≥ t. So với sự thật từ gold
(bên có F1 span cao hơn) để ra precision/recall. Ngưỡng đề xuất là t nhỏ nhất đạt ``--precision``
với ít nhất ``--min-support`` case được xác nhận.

Mặc định chỉ in bảng. ``--write`` mới ghi ``gate.min_severity`` vào thresholds.yaml và tăng ``version``;
các ngưỡng ``retrain.*`` vẫn phải do người quyết định sau khi xem metrics hiệu chuẩn.

  .venv-layer1/Scripts/python.exe -m feedback.eval.calibrate <calibration_run_id> [--precision 0.8] [--write]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import yaml

from feedback.core.config import CONFIG_DIR
from feedback.core.schemas import CaseRecord, Decision
from feedback.eval.report import truth_winner


def sweep(records: list[CaseRecord]) -> list[dict[str, Any]]:
    labeled = [(r, truth_winner(r)) for r in records if r.meta.get("gold") is not None and r.new is not None]
    positives = sum(1 for _, truth in labeled if truth == "deepseek")
    candidates = [(r.diff.get("severity_total", 0.0), truth) for r, truth in labeled
                  if r.decision is Decision.DEEPSEEK_BETTER]
    rows = []
    for t in sorted({0.0} | {s for s, _ in candidates}):
        kept = [truth for s, truth in candidates if s >= t]
        tp = sum(truth == "deepseek" for truth in kept)
        rows.append({"min_severity": t, "confirmed": len(kept), "tp": tp,
                     "precision": tp / len(kept) if kept else None,
                     "recall": tp / positives if positives else None})
    return rows


def suggest(rows: list[dict[str, Any]], precision: float, min_support: int) -> float | None:
    for row in rows:
        if row["precision"] is not None and row["precision"] >= precision and row["confirmed"] >= min_support:
            return row["min_severity"]
    return None


def main() -> int:
    from feedback.pipeline.runner import Pipeline

    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--precision", type=float, default=0.8)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    pipeline = Pipeline()
    run = pipeline.store.get_run(args.run_id)
    if run is None or run["kind"] != "calibration":
        sys.exit("Cần run_id của một lượt hiệu chuẩn")
    rows = sweep(pipeline.store.records(args.run_id))
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    value = suggest(rows, args.precision, args.min_support)
    print(f"đề xuất gate.min_severity = {value}  (precision ≥ {args.precision}, ≥ {args.min_support} case)")
    if args.write and value is not None:
        path = CONFIG_DIR / "thresholds.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["gate"]["min_severity"] = value
        data["version"] = int(data.get("version") or 0) + 1
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"đã ghi {path} (version {data['version']}) — chú thích trong file bị mất khi ghi lại, xem git diff")
    return 0


if __name__ == "__main__":
    sys.exit(main())
