"""Đo tham số lấy mẫu của Qwen judge trên bộ probe trước khi chốt vào models.yaml (BUILD_PIPELINE.md §6.4).

  .venv-layer1/Scripts/python.exe -m feedback.scripts.measure_judge_params [--budget 1.5] [--workers 4]

So từng cấu hình theo: đúng trên probe known/identical/verbose, tỉ lệ lật, số lần phải retry
(dấu hiệu kẹt vòng lặp suy luận), reasoning token, độ trễ và chi phí. Ghi kết quả vào
golden_dataset/evals/judge_params/results.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from feedback.core import llm_client
from feedback.core.config import PROJECT_ROOT
from feedback.eval import probes as probe_mod
from feedback.stages.qwen_judge import judge_pair, system_prompt

CONFIGS = {
    "hien_tai": {},                                                          # temperature 0 trong models.yaml
    "khuyen_nghi": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5},
}
OUT = PROJECT_ROOT / "golden_dataset" / "evals" / "judge_params"


class CapBudget:
    def __init__(self, limit: float) -> None:
        self.limit, self.spent, self.reserved, self.lock = limit, 0.0, 0.0, threading.Lock()

    def reserve(self, estimate: float) -> bool:
        with self.lock:
            if self.spent + self.reserved + estimate > self.limit:
                return False
            self.reserved += estimate
            return True

    def settle(self, estimate: float, actual: float) -> None:
        with self.lock:
            self.reserved -= estimate
            self.spent += actual


def run_one(probe: probe_mod.Probe, overrides: dict[str, Any], budget: CapBudget, system: str) -> dict[str, Any]:
    winners, calls = [], []
    for first_is_good in (True, False):
        first, second = (probe.good, probe.other) if first_is_good else (probe.other, probe.good)
        try:
            raw, call = judge_pair(probe.text, first, second, budget, system, overrides)
        except llm_client.LLMError as exc:
            calls.append({"error": str(exc), "cost_usd": exc.cost_usd, "attempts": exc.attempts})
            winners.append("error")
            continue
        side = "tie" if raw == "tie" else ("good" if (raw == "1") == first_is_good else "other")
        winners.append(side)
        calls.append({"raw": raw, "cost_usd": call.cost_usd, "elapsed_sec": call.elapsed_sec,
                      "reasoning": call.usage["reasoning"], "attempts": len(call.attempts),
                      "issues": {k: call.parsed.get(k) for k in ("issues_1", "issues_2")}})
    return {**probe_mod.score(probe, winners), "calls": calls, "source": probe.source}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [c for r in rows for c in r["calls"] if "error" not in c]
    by_kind = {}
    for kind in ("known", "identical", "verbose"):
        sub = [r for r in rows if r["kind"] == kind]
        by_kind[kind] = {"n": len(sub), "correct": sum(r["correct"] for r in sub),
                         "flip": sum(not r["consistent"] for r in sub)}
    ident_calls = [c for r in rows if r["kind"] == "identical" for c in r["calls"] if "raw" in c]
    lat = sorted(c["elapsed_sec"] for c in calls)
    return {
        "by_kind": by_kind,
        "flip_rate": sum(not r["consistent"] for r in rows) / len(rows) if rows else None,
        "identical_slot_picks": {k: sum(c["raw"] == k for c in ident_calls) for k in ("1", "2", "tie")},
        "errors": sum(1 for r in rows for c in r["calls"] if "error" in c),
        "calls_with_retry": sum(1 for c in calls if c["attempts"] > 1),
        "reasoning_mean": round(statistics.fmean(c["reasoning"] for c in calls)) if calls else None,
        "latency_mean": round(statistics.fmean(lat), 1) if lat else None,
        "latency_p95": lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))] if lat else None,
        "cost_usd": round(sum(c["cost_usd"] for r in rows for c in r["calls"]), 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    probes = probe_mod.build(PROJECT_ROOT / "golden_dataset" / "evals" / "golden_test2",
                             PROJECT_ROOT / "golden_dataset" / "golden_test2.json")
    print(f"{len(probes)} probe × 2 thứ tự × {len(CONFIGS)} cấu hình = {len(probes) * 2 * len(CONFIGS)} lần gọi", flush=True)
    system, shas = system_prompt()
    budget = CapBudget(args.budget)
    results: dict[str, Any] = {"shas": shas, "configs": CONFIGS, "n_probes": len(probes), "runs": {}}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, overrides in CONFIGS.items():
        started = time.monotonic()
        with ThreadPoolExecutor(args.workers) as pool:
            rows = list(pool.map(lambda p: run_one(p, overrides, budget, system), probes))
        summary = summarize(rows)
        summary["wall_sec"] = round(time.monotonic() - started)
        results["runs"][name] = {"summary": summary, "rows": rows}
        print(f"[{name}] {json.dumps(summary, ensure_ascii=False)}", flush=True)
        (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"tổng chi phí ${budget.spent:.4f} · ghi {OUT / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
