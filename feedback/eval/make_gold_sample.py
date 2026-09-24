"""Xuất hàng chờ gán nhãn tay từ một run: case random audit + case judge lật (inconclusive).

File xuất ra theo định dạng golden_dataset (``{"texts": [...], "results": [{"result": {...}}]}``) với
``result`` để TRỐNG: người gán nhãn điền L1–L7, rồi dùng file đó làm gold cho một lượt hiệu chuẩn.
Không kèm bản parse của model hay DeepSeek — nhìn thấy một bản làm sẵn sẽ kéo nhãn tay về phía bản đó
và làm gold nghiêng, khiến judge trông đúng hơn thực tế.

  .venv-layer1/Scripts/python.exe -m feedback.eval.make_gold_sample <run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from feedback.core.config import PROJECT_ROOT
from feedback.core.schemas import CaseRecord, Decision

HUMAN_QUEUE = PROJECT_ROOT / "feedback" / "store" / "human_queue"


def queue_items(records: list[CaseRecord]) -> list[tuple[CaseRecord, str]]:
    items = []
    for record in records:
        if record.meta.get("gold") is not None:
            continue                                  # đã có nhãn
        if record.meta.get("audit"):
            items.append((record, "random_audit"))
        elif record.decision is Decision.INCONCLUSIVE:
            items.append((record, "judge_flip"))
    return items


def to_label_file(run_id: str, records: list[CaseRecord]) -> dict[str, Any]:
    items = queue_items(records)
    empty = {f"L{i}": [] for i in range(1, 8)}
    return {
        "_meta": {"run_id": run_id, "purpose": "gán nhãn tay cho hiệu chuẩn judge",
                  "instructions": "Điền L1–L7 cho từng text theo prompt/v2; để trống level không có."},
        "texts": [r.raw_text for r, _ in items],
        "results": [{"result": {"input": r.raw_text, **empty}, "case_id": r.case_id, "reason": reason}
                    for r, reason in items],
    }


def main() -> int:
    from feedback.pipeline.runner import Pipeline

    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    pipeline = Pipeline()
    payload = to_label_file(args.run_id, pipeline.store.records(args.run_id))
    HUMAN_QUEUE.mkdir(parents=True, exist_ok=True)
    out = HUMAN_QUEUE / f"{args.run_id}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(payload['texts'])} case → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
