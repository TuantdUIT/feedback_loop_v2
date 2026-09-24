"""Tổng hợp metrics của một run và khuyến nghị retrain (BUILD_PIPELINE.md §7.2–7.4).

Module chỉ đọc ``CaseRecord`` đã xong, không sửa dữ liệu. Ba nhóm:
  - metrics của run: luồng, cost, đồng thuận model ↔ DeepSeek, sức khoẻ judge, ước lượng lỗi model;
  - hiệu chuẩn (chỉ khi case có gold): judge chọn đúng tới đâu so với sự thật;
  - khuyến nghị retrain: chỉ bật khi judge đã hiệu chuẩn cho ĐÚNG cấu hình hiện tại và mọi ngưỡng có giá trị.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from typing import Any

from feedback.core.schemas import CaseRecord, Decision

LEVELS = [f"L{i}" for i in range(1, 8)]
JUDGED = {Decision.MODEL_BETTER, Decision.DEEPSEEK_BETTER, Decision.TIE, Decision.INCONCLUSIVE}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Khoảng tin cậy Wilson 95% cho tỉ lệ k/n."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": wilson(k, n)}


def _latency(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    v = sorted(values)
    return {"mean": round(statistics.fmean(v), 2), "p50": v[len(v) // 2],
            "p95": v[min(len(v) - 1, round(0.95 * (len(v) - 1)))], "max": v[-1]}


def _prf(tp: int, n_pred: int, n_ref: int) -> dict[str, Any]:
    p = tp / n_pred if n_pred else None
    r = tp / n_ref if n_ref else None
    f = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "n_pred": n_pred, "n_ref": n_ref}


# ----------------------------------------------------------------- run metrics
def run_metrics(records: list[CaseRecord]) -> dict[str, Any]:
    decisions = Counter(r.decision.value if r.decision else "open" for r in records)
    compared = [r for r in records if r.new is not None and r.diff]

    # đồng thuận span, coi DeepSeek là mốc so (không phải đáp án đúng)
    tp = sum(r.diff["counts"]["COR"] for r in compared)
    per_level = {}
    for lv in LEVELS:
        lv_tp = sum(1 for r in compared for i in r.diff["items"] if i["kind"] == "COR" and i["old"]["level"] == lv)
        lv_old = sum(1 for r in compared for s in r.old.spans if s.level == lv)
        lv_new = sum(1 for r in compared for s in r.new.spans if s.level == lv)
        if lv_old or lv_new:
            per_level[lv] = _prf(lv_tp, lv_old, lv_new)

    judged = [r for r in records if r.decision in JUDGED]
    calls = [c for r in judged for c in r.layers.get("judge", {}).get("calls", [])]
    non_tie_calls = [c for c in calls if c["raw_winner"] in ("1", "2")]
    decided = [r for r in judged if r.decision in (Decision.MODEL_BETTER, Decision.DEEPSEEK_BETTER)]

    def more_spans(r: CaseRecord, side: str) -> bool:
        other = len(r.new.spans) if side == "model" else len(r.old.spans)
        mine = len(r.old.spans) if side == "model" else len(r.new.spans)
        return mine > other

    winner_side = {Decision.MODEL_BETTER: "model", Decision.DEEPSEEK_BETTER: "deepseek"}
    unequal = [r for r in decided if len(r.old.spans) != len(r.new.spans)]
    ds_better = [r for r in records if r.decision is Decision.DEEPSEEK_BETTER]
    confirmed = [r for r in ds_better if r.layers.get("gate", {}).get("confirmed_error")]
    evaluable = [r for r in records if r.decision in JUDGED | {Decision.AGREE}]

    kinds = Counter(i["kind"] for r in ds_better for i in r.diff["items"] if i["kind"] != "COR")
    levels = Counter(lv for r in ds_better for lv in r.diff.get("levels_involved", []))

    ds_cost = sum(c["cost_usd"] for r in records for c in r.layers.get("deepseek", {}).get("calls", []))
    judge_cost = sum(c["cost_usd"] for c in calls)
    return {
        "n_cases": len(records),
        "decisions": dict(decisions),
        "cost": {"deepseek_usd": round(ds_cost, 8), "judge_usd": round(judge_cost, 8),
                 "cases_total_usd": round(sum(r.cost_usd for r in records), 8)},
        "latency_sec": {
            "deepseek": _latency([c["elapsed_sec"] for r in records for c in r.layers.get("deepseek", {}).get("calls", [])]),
            "judge": _latency([c["elapsed_sec"] for c in calls]),
        },
        "agreement": {
            "identical": _rate(sum(1 for r in compared if r.diff["identical"]), len(compared)),
            "span_model_vs_deepseek": _prf(tp, sum(r.diff["n_old"] for r in compared),
                                           sum(r.diff["n_new"] for r in compared)),
            "per_level": per_level,
        },
        "judge_health": {
            "n_judged": len(judged),
            "flip_rate": _rate(sum(1 for r in judged if r.decision is Decision.INCONCLUSIVE), len(judged)),
            "first_slot_win_rate": _rate(sum(1 for c in non_tie_calls if c["raw_winner"] == "1"), len(non_tie_calls)),
            "tie_rate": _rate(sum(1 for r in judged if r.decision is Decision.TIE), len(judged)),
            "winner_has_more_spans_rate": _rate(sum(1 for r in unequal if more_spans(r, winner_side[r.decision])),
                                                len(unequal)),
            "verdicts": dict(Counter(r.decision.value for r in judged)),
        },
        "model_quality": {
            "deepseek_better": _rate(len(ds_better), len(evaluable)),
            "confirmed_error": _rate(len(confirmed), len(evaluable)),
            "error_kinds": dict(kinds),
            "levels_involved": dict(levels),
            "by_level_rate": {lv: _rate(levels[lv], len(evaluable)) for lv in LEVELS if levels[lv]},
        },
        "audit": {"n_flagged": sum(1 for r in records if r.meta.get("audit"))},
    }


# ----------------------------------------------------------------- calibration
def _norm(text: str) -> str:
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).lower())
    return text.strip(" ,.;:-")


def _gold_counter(gold: dict[str, Any]) -> Counter:
    return Counter((lv, _norm(t)) for lv in LEVELS for t in gold.get(lv, []) if _norm(t))


def _span_counter(spans: list) -> Counter:
    return Counter((s.level, _norm(s.text)) for s in spans if _norm(s.text))


def _f1(pred: Counter, gold: Counter) -> float:
    tp = sum((pred & gold).values())
    n_pred, n_gold = sum(pred.values()), sum(gold.values())
    if n_pred == 0 and n_gold == 0:
        return 1.0
    return 2 * tp / (n_pred + n_gold) if n_pred + n_gold else 0.0


def truth_winner(record: CaseRecord) -> str | None:
    """Bên thật sự tốt hơn theo gold: F1 span cao hơn; bằng nhau → 'tie'."""
    gold = record.meta.get("gold")
    if gold is None or record.new is None:
        return None
    g = _gold_counter(gold)
    fm, fd = _f1(_span_counter(record.old.spans), g), _f1(_span_counter(record.new.spans), g)
    if abs(fm - fd) < 1e-9:
        return "tie"
    return "model" if fm > fd else "deepseek"


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[x] * cb[x] for x in labels) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def calibration_metrics(records: list[CaseRecord]) -> dict[str, Any] | None:
    """Judge chọn đúng tới đâu, trên các case có gold. None nếu không có case nào có gold."""
    with_gold = [r for r in records if r.meta.get("gold") is not None and r.new is not None]
    if not with_gold:
        return None
    to_label = {Decision.MODEL_BETTER: "model", Decision.DEEPSEEK_BETTER: "deepseek", Decision.TIE: "tie"}
    judged = [(to_label[r.decision], truth_winner(r)) for r in with_gold if r.decision in to_label]
    predicted_ds = [t for p, t in judged if p == "deepseek"]
    truth_ds = [p for p, t in judged if t == "deepseek"]
    not_ds = [p for p, t in judged if t != "deepseek"]
    agreed = [r for r in with_gold if r.decision is Decision.AGREE]
    both_wrong = [r for r in agreed if _f1(_span_counter(r.old.spans), _gold_counter(r.meta["gold"])) < 1.0]
    flips = sum(1 for r in with_gold if r.decision is Decision.INCONCLUSIVE)
    n_judge = len(judged) + flips
    return {
        "n_with_gold": len(with_gold),
        "n_judged": len(judged),
        "accuracy": _rate(sum(p == t for p, t in judged), len(judged)),
        "kappa": cohen_kappa(judged),
        "detect_precision": _rate(sum(t == "deepseek" for t in predicted_ds), len(predicted_ds)),
        "detect_recall": _rate(sum(p == "deepseek" for p in truth_ds), len(truth_ds)),
        "false_alarm": _rate(sum(p == "deepseek" for p in not_ds), len(not_ds)),
        "flip_rate": _rate(flips, n_judge),
        "agree_but_wrong": _rate(len(both_wrong), len(agreed)),
        "confusion": dict(Counter(f"{p}|{t}" for p, t in judged)),
    }


# ----------------------------------------------------------------- recommendation
def recommendation(metrics: dict[str, Any], thresholds: dict[str, Any],
                   calibration: dict[str, Any] | None) -> dict[str, Any]:
    retrain = thresholds.get("retrain") or {}
    min_severity = (thresholds.get("gate") or {}).get("min_severity")
    needed = {"gate.min_severity": min_severity, **{f"retrain.{k}": retrain.get(k)
              for k in ("min_kappa", "max_false_alarm", "max_flip_rate", "min_cases", "max_error_rate")}}
    missing = [k for k, v in needed.items() if v is None]
    if calibration is None or missing:
        reasons = []
        if calibration is None:
            reasons.append("Chưa có lượt hiệu chuẩn nào cho cấu hình judge hiện tại")
        if missing:
            reasons.append("Ngưỡng chưa được hiệu chuẩn: " + ", ".join(missing))
        return {"status": "uncalibrated", "label": "Chưa hiệu chuẩn — chưa có căn cứ khuyến nghị retrain",
                "reasons": reasons}

    cal = calibration["metrics"]
    reasons, judge_ok = [], True
    kappa = cal.get("kappa")
    if kappa is None or kappa < retrain["min_kappa"]:
        judge_ok = False
        reasons.append(f"Judge chưa đạt chuẩn: κ = {kappa} < {retrain['min_kappa']}")
    false_alarm = (cal.get("false_alarm") or {}).get("rate")
    if false_alarm is None or false_alarm > retrain["max_false_alarm"]:
        judge_ok = False
        reasons.append(f"Tỉ lệ báo động giả {false_alarm} > {retrain['max_false_alarm']}")
    flip = metrics["judge_health"]["flip_rate"]["rate"]
    if flip is not None and flip > retrain["max_flip_rate"]:
        judge_ok = False
        reasons.append(f"Tỉ lệ lật {flip:.2f} > {retrain['max_flip_rate']}")
    if not judge_ok:
        return {"status": "judge_unreliable", "label": "Judge chưa đủ tin cậy — không khuyến nghị", "reasons": reasons}

    confirmed = metrics["model_quality"]["confirmed_error"]
    precision = (cal.get("detect_precision") or {}).get("rate")
    low = (confirmed["ci95"] or (0, 0))[0]
    triggered = confirmed["k"] >= retrain["min_cases"] and low >= retrain["max_error_rate"]
    slices = [lv for lv, rate in metrics["model_quality"]["by_level_rate"].items()
              if rate["k"] >= retrain["min_cases"] and (rate["ci95"] or (0, 0))[0] >= retrain["max_error_rate"]]
    return {
        "status": "retrain" if triggered or slices else "no_retrain",
        "label": "Khuyến nghị retrain" if triggered or slices else "Chưa cần retrain",
        "reasons": reasons or [f"{confirmed['k']} lỗi đã xác nhận / {confirmed['n']} case, cận dưới CI95 {low:.3f}"],
        "slices": slices,
        "corrected_error_rate": confirmed["rate"] * precision if confirmed["rate"] is not None and precision else None,
    }


def build_report(records: list[CaseRecord], thresholds: dict[str, Any],
                 calibration: dict[str, Any] | None) -> dict[str, Any]:
    metrics = run_metrics(records)
    return {"metrics": metrics, "calibration_run": calibration_metrics(records),
            "calibration_used": ({"run_id": calibration["run_id"], "created_at": calibration["created_at"]}
                                 if calibration else None),
            "recommendation": recommendation(metrics, thresholds, calibration)}
