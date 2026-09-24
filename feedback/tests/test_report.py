"""Test cho eval.report, eval.calibrate, eval.make_gold_sample và scripts.export_silver."""

import pytest

from feedback.core.diff_metrics import compare
from feedback.core.schemas import CaseRecord, Decision, ParseResult, Span
from feedback.eval.calibrate import suggest, sweep
from feedback.eval.make_gold_sample import to_label_file
from feedback.eval.report import build_report, cohen_kappa, recommendation, truth_winner, wilson
from feedback.scripts.export_silver import silver_rows


def record(text: str, old: list[Span], new: list[Span], decision: Decision, gold=None, audit=False,
           confirmed=None) -> CaseRecord:
    r = CaseRecord(case_id=f"t#{abs(hash(text)) % 99999:05d}", raw_text=text,
                   old=ParseResult([], [], old, "gsm_api"), meta={"audit": audit},
                   new=ParseResult([], [], new, "deepseek"))
    r.diff = compare(old, new)
    r.decision = decision
    r.layers["gate"] = {"confirmed_error": confirmed, "severity_total": r.diff["severity_total"]}
    if gold is not None:
        r.meta["gold"] = gold
    return r


WARD = [Span(0, 8, "L4", "Bến Nghé"), Span(9, 15, "L3", "Quận 1")]
MERGED = [Span(0, 15, "L4", "Bến Nghé Quận 1")]


def test_wilson_bounds() -> None:
    low, high = wilson(5, 10)
    assert 0.23 < low < 0.24 and 0.76 < high < 0.77
    assert wilson(0, 0) is None


def test_kappa_perfect_and_chance() -> None:
    assert cohen_kappa([("a", "a"), ("b", "b")]) == pytest.approx(1.0)
    assert cohen_kappa([("a", "b"), ("b", "a")]) == pytest.approx(-1.0)


def test_truth_winner_from_gold() -> None:
    gold = {"L4": ["Bến Nghé"], "L3": ["Quận 1"]}
    r = record("Bến Nghé Quận 1", MERGED, WARD, Decision.DEEPSEEK_BETTER, gold=gold)
    assert truth_winner(r) == "deepseek"


def test_uncalibrated_blocks_recommendation() -> None:
    r = record("Bến Nghé Quận 1", MERGED, WARD, Decision.DEEPSEEK_BETTER)
    report = build_report([r], {"gate": {"min_severity": None}, "retrain": {"min_kappa": 0.6}}, None)
    assert report["recommendation"]["status"] == "uncalibrated"
    assert report["metrics"]["model_quality"]["deepseek_better"]["k"] == 1


THRESHOLDS = {"gate": {"min_severity": 0.5},
              "retrain": {"min_kappa": 0.6, "max_false_alarm": 0.1, "max_flip_rate": 0.2,
                          "min_cases": 2, "max_error_rate": 0.1}}


def test_recommendation_retrain_when_judge_ok_and_errors_high() -> None:
    records = [record(f"Bến Nghé Quận {i}", MERGED, WARD, Decision.DEEPSEEK_BETTER, confirmed=True) for i in range(8)]
    metrics = build_report(records, THRESHOLDS, None)["metrics"]
    good_judge = {"run_id": "cal", "created_at": 0, "metrics": {"kappa": 0.8, "false_alarm": {"rate": 0.05},
                                                               "detect_precision": {"rate": 0.9}}}
    rec = recommendation(metrics, THRESHOLDS, good_judge)
    assert rec["status"] == "retrain"
    assert rec["corrected_error_rate"] == pytest.approx(0.9)


def test_recommendation_blocked_by_bad_judge() -> None:
    records = [record("x", MERGED, WARD, Decision.DEEPSEEK_BETTER, confirmed=True)]
    metrics = build_report(records, THRESHOLDS, None)["metrics"]
    bad = {"run_id": "cal", "created_at": 0, "metrics": {"kappa": 0.3, "false_alarm": {"rate": 0.05}}}
    assert recommendation(metrics, THRESHOLDS, bad)["status"] == "judge_unreliable"


def test_calibration_sweep_and_suggest() -> None:
    gold = {"L4": ["Bến Nghé"], "L3": ["Quận 1"]}
    right = record("Bến Nghé Quận 1", MERGED, WARD, Decision.DEEPSEEK_BETTER, gold=gold)
    wrong = record("Bến Nghé Quận 1 ", WARD, MERGED, Decision.DEEPSEEK_BETTER,
                   gold={"L4": ["Bến Nghé"], "L3": ["Quận 1"]})
    rows = sweep([right, wrong])
    assert rows[0]["confirmed"] == 2 and rows[0]["precision"] == 0.5
    assert suggest(rows, precision=0.9, min_support=1) is None      # không ngưỡng nào đạt 90%
    assert suggest(rows, precision=0.5, min_support=2) == 0.0


def test_label_queue_has_text_only() -> None:
    audit = record("Quận 1", [Span(0, 6, "L3", "Quận 1")], [Span(0, 6, "L3", "Quận 1")], Decision.AGREE, audit=True)
    flip = record("Tân Phú", [Span(0, 7, "L4", "Tân Phú")], [Span(0, 7, "L3", "Tân Phú")], Decision.INCONCLUSIVE)
    other = record("Bà Triệu", [], [], Decision.AGREE)
    payload = to_label_file("run-x", [audit, flip, other])
    assert payload["texts"] == ["Quận 1", "Tân Phú"]
    assert [r["reason"] for r in payload["results"]] == ["random_audit", "judge_flip"]
    assert all(not any(r["result"][f"L{i}"] for i in range(1, 8)) for r in payload["results"])


def test_silver_only_confirmed_and_never_calibration() -> None:
    confirmed = record("a", MERGED, WARD, Decision.DEEPSEEK_BETTER, confirmed=True)
    candidate = record("b", MERGED, WARD, Decision.DEEPSEEK_BETTER, confirmed=None)
    assert len(silver_rows({"kind": "normal"}, [confirmed, candidate])) == 1
    assert silver_rows({"kind": "calibration"}, [confirmed]) == []
