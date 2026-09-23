"""Đánh giá LLM qua API (Qwen/OpenRouter, DeepSeek) trên các bộ golden test bằng các hàm của eval_golden.

Đầu vào là file `.jsonl` do `scripts/external_api/*/run_*_on_file.py` ghi ra, chạy trên
`golden_dataset/evals/<test>/input.txt` (tạo bởi `scripts/make_golden_test.py`). Output của
model (`spans[].level`, `spans[].text`) so với gold bằng đúng `evaluate()` của `eval_golden.py`:
Accuracy, Precision/Recall/F1 micro và theo level, Latency. Mục cuối là cost tổng.

Latency là thời gian gọi API end-to-end của một mẫu, gồm cả các lần retry.
Mẫu không trả được JSON có `spans` bị tính là dự đoán rỗng (không bỏ khỏi mẫu số).

Cost:
  qwen      `usage.cost` do OpenRouter trả, đã cộng mọi lần thử.
  deepseek  API không trả chi phí -> tính từ token từng lần thử theo bảng giá DEEPSEEK_PRICE,
            chọn giá cao/thấp điểm theo thời điểm gửi (UTC) ghi trong `attempts[].utc`.

Chạy (cần numpy vì dùng lại eval_golden), --test mặc định golden_test:
  .venv-layer1/Scripts/python.exe scripts/eval_golden_llm.py [--test golden_test2]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_golden import LEVELS, evaluate, gold_spans, to_counter  # noqa: E402

GOLDEN_DIR = ROOT / "golden_dataset"
PROVIDERS = {"qwen": "OPENROUTER_MODEL", "deepseek": "DEEPSEEK_MODEL"}

# USD / 1M token, giá thấp điểm; cao điểm gấp đôi. Nguồn: api-docs.deepseek.com/quick_start/pricing
# (xem ngày 2026-09-23). Cao điểm: thứ Hai–thứ Sáu, 01:00–04:00 và 06:00–10:00 UTC.
DEEPSEEK_PRICE = {
    "deepseek-flash": {"miss": 0.15, "hit": 0.003, "out": 0.6},
    "deepseek-v4-pro": {"miss": 0.66, "hit": 0.022, "out": 1.98},
}
DEEPSEEK_PEAK_HOURS = [(1, 4), (6, 10)]


def read_env() -> dict[str, str]:
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k] = v.strip().strip('"').strip("'")
    return env


def is_peak(utc: str) -> bool:
    t = time.strptime(utc, "%Y-%m-%dT%H:%M:%SZ")
    return t.tm_wday < 5 and any(a <= t.tm_hour < b for a, b in DEEPSEEK_PEAK_HOURS)


def pred_spans(parsed) -> list[tuple[str, str]] | None:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("spans"), list):
        return None
    return [(s["level"], s["text"]) for s in parsed["spans"]
            if isinstance(s, dict) and s.get("level") in LEVELS and isinstance(s.get("text"), str)]


def cost_qwen(records: list[dict]) -> dict:
    return {
        "total_usd": sum(r["usage"].get("cost", 0) or 0 for r in records),
        "source": "OpenRouter usage.cost",
        "prompt_tokens": sum(r["usage"].get("prompt_tokens", 0) or 0 for r in records),
        "completion_tokens": sum(r["usage"].get("completion_tokens", 0) or 0 for r in records),
        "reasoning_tokens": sum((r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
                                for r in records),
    }


def cost_deepseek(records: list[dict], model: str) -> dict:
    price = DEEPSEEK_PRICE[model]
    tok = {"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    total = 0.0
    peak_calls = calls = 0
    for r in records:
        for a in r["attempts"]:
            if a.get("completion_tokens") is None:      # lỗi HTTP/mạng: không có usage, không tính phí
                continue
            calls += 1
            mult = 2 if is_peak(a["utc"]) else 1
            peak_calls += mult == 2
            hit, miss, out = (a.get(k) or 0 for k in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                                                      "completion_tokens"))
            total += mult * (miss * price["miss"] + hit * price["hit"] + out * price["out"]) / 1e6
            for k in tok:
                tok[k] += a.get(k) or 0
    return {"total_usd": total, "source": f"tính từ token × giá {model} (thấp điểm {price}, cao điểm ×2)",
            "billed_calls": calls, "peak_calls": peak_calls, **tok}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="golden_test", help="tên bộ test: golden_dataset/<test>.json, evals/<test>/")
    args = ap.parse_args()
    GOLDEN_TEST = GOLDEN_DIR / f"{args.test}.json"
    OUT_DIR = GOLDEN_DIR / "evals" / args.test

    env = read_env()
    golden = json.loads(GOLDEN_TEST.read_text(encoding="utf-8"))
    samples = list(zip(golden["texts"], (r["result"] for r in golden["results"]), golden["sources"], strict=True))
    subsets = sorted({s["subset"] for _, _, s in samples})

    metrics: dict = {"dataset": {"file": GOLDEN_TEST.name, "seed": golden.get("seed"), "n": len(samples),
                                 "subsets": {k: sum(s["subset"] == k for _, _, s in samples) for k in subsets}},
                     "models": {}}
    for provider, model_key in PROVIDERS.items():
        raw_path = OUT_DIR / f"raw_{provider}.jsonl"
        if not raw_path.exists():
            print(f"[{provider}] bỏ qua: chưa có {raw_path.relative_to(ROOT)}")
            continue
        records = {r["i"]: r for r in map(json.loads, raw_path.read_text(encoding="utf-8").splitlines()) if r}
        missing = [i for i in range(len(samples)) if i not in records]
        if missing:
            print(f"[{provider}] bỏ qua: thiếu {len(missing)} mẫu {missing[:5]}… — chạy lại script API")
            continue

        rows = []
        for i, (text, gold, src) in enumerate(samples):
            rec = records[i]
            assert rec["text"] == text, f"mẫu {i} lệch input: {rec['text']!r} != {text!r}"
            pred = pred_spans(rec["parsed"])
            rows.append({"i": i, "subset": src["subset"], "text": text, "gold": gold_spans(gold),
                         "pred": pred or [], "parse_ok": pred is not None,
                         "latency_ms": {"total": rec["sec"] * 1000}})
        for r in rows:
            r["exact"] = to_counter(r["gold"]) == to_counter(r["pred"])
        with (OUT_DIR / f"pred_{provider}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        model = env.get(model_key, "?")
        cost = cost_qwen(list(records.values())) if provider == "qwen" else cost_deepseek(list(records.values()), model)
        cost["per_sample_usd"] = cost["total_usd"] / len(rows)
        metrics["models"][provider] = {
            "model": model,
            "parse_ok": sum(r["parse_ok"] for r in rows),
            "retried": sum(1 for r in records.values() if r["usage"].get("retries")),
            "results": {"all": evaluate(rows), **{k: evaluate([r for r in rows if r["subset"] == k]) for k in subsets}},
            "cost": cost,                                   # mục cuối: cost tổng sau khi chạy
        }

    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    for provider, m in metrics["models"].items():
        print(f"\n[{provider}] {m['model']} | parse OK {m['parse_ok']}/{metrics['dataset']['n']} | retry {m['retried']}")
        for name, res in m["results"].items():
            lat = res["latency_ms"]["total"]
            print(f"  {name:<10} n={res['n']:<3} acc {res['accuracy']:.3f} ({res['exact_match']}/{res['n']}) | "
                  f"P {res['micro']['precision']:.3f} R {res['micro']['recall']:.3f} F1 {res['micro']['f1']:.3f} | "
                  f"latency mean {lat['mean'] / 1000:.1f}s p95 {lat['p95'] / 1000:.1f}s")
        per_lv = m["results"]["all"]["per_level"]
        print("  F1 theo level: " + ", ".join(f"{lv} {v['f1']:.2f}" for lv, v in per_lv.items()))
        c = m["cost"]
        print(f"  COST tổng ${c['total_usd']:.5f} | ${c['per_sample_usd']:.5f}/mẫu | {c['source']}")
    print(f"\nđã ghi {(OUT_DIR / 'metrics.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
