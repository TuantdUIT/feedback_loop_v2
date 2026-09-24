"""Chạy pipeline trên một file capture mà không cần giao diện; ghi vào cùng DB nên run hiện cả trên UI.

  .venv-layer1/Scripts/python.exe -m feedback.scripts.run_pipeline kafka_simulation/template.txt
  .venv-layer1/Scripts/python.exe -m feedback.scripts.run_pipeline cap.txt --calibration gold.json
  .venv-layer1/Scripts/python.exe -m feedback.scripts.run_pipeline --preview kafka_simulation/template.txt

Đừng chạy song song với server đang bật trên cùng DB: hai tiến trình sẽ cùng giành message. Khi server
đang chạy, upload qua giao diện thay vì dùng script này.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from feedback.pipeline.runner import DEFAULT_DB, Pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("capture", type=Path)
    parser.add_argument("--calibration", type=Path, help="file gold (định dạng golden_dataset) → lượt hiệu chuẩn")
    parser.add_argument("--preview", action="store_true", help="chỉ đọc file và ước tính chi phí")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    pipeline = Pipeline(args.db)
    content = args.capture.read_text(encoding="utf-8")
    if args.preview:
        print(json.dumps(pipeline.preview(args.capture.name, content), ensure_ascii=False, indent=2))
        return 0

    gold = json.loads(args.calibration.read_text(encoding="utf-8")) if args.calibration else None
    pipeline.start()
    try:
        run_id = pipeline.submit(args.capture.name, content, "calibration" if gold else "normal", gold)
        print(f"run {run_id}: {pipeline.store.get_run(run_id)['n_cases']} case", flush=True)
        run = pipeline.wait(run_id, timeout=args.timeout)
    finally:
        pipeline.stop()
    metrics = run["report"]["metrics"]
    print(json.dumps({"run_id": run_id, "cost_usd": run["cost_usd"], "decisions": metrics["decisions"],
                      "cost": metrics["cost"], "judge_health": metrics["judge_health"],
                      "recommendation": run["report"]["recommendation"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
