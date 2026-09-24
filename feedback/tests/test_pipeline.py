"""Test đầu-cuối của pipeline với mock API: trigger dây chuyền, quyết định, cost, ngân sách, lỗi."""

from __future__ import annotations

import json

import pytest

from feedback.core.schemas import Decision
from feedback.pipeline.runner import Pipeline
from feedback.tests.conftest import chat_response

# text -> (spans model trả trong capture, spans DeepSeek sẽ trả)
CASES = {
    "Quận 1, Hồ Chí Minh": ({"L3": ["Quận 1"], "L2": ["Hồ Chí Minh"]}, [("L3", "Quận 1"), ("L2", "Hồ Chí Minh")]),
    "Bến NghéQuận 1": ({"L4": ["Bến NghéQuận 1"]}, [("L4", "Bến Nghé"), ("L3", "Quận 1")]),
    "Tân Phú Hồ Chí Minh": ({"L3": ["Tân Phú Hồ Chí Minh"]}, [("L3", "Tân Phú"), ("L2", "Hồ Chí Minh")]),
    "74 Nơ Trang Long": ({"L6": ["74"], "L5": ["Nơ Trang Long"]}, [("L6", "74"), ("L4", "Nơ Trang Long")]),
    "Hỏng offset": ({"L5": ["Hỏng offset"]}, "BROKEN"),
}
# text -> judge luôn chọn DeepSeek / model / lật theo vị trí
JUDGE = {"Bến NghéQuận 1": "deepseek", "Tân Phú Hồ Chí Minh": "flip", "74 Nơ Trang Long": "model"}


def capture(texts: list[str]) -> str:
    results = []
    for t in texts:
        result = {f"L{i}": [] for i in range(1, 8)}
        result.update(CASES[t][0])
        results.append({"result": {"input": t, **result}})
    request = json.dumps({"texts": texts}, ensure_ascii=False)
    return (f"curl --location 'http://x.test/ner/predict-batch' --data '{request}'\n"
            + json.dumps({"results": results}, ensure_ascii=False))


def ds_spans(text: str) -> list[dict]:
    spec = CASES[text][1]
    if spec == "BROKEN":
        return [{"level": "L5", "start": 0, "end": 3, "text": "không có trong input"}]
    return [{"level": lv, "start": text.index(t), "end": text.index(t) + len(t), "text": t, "truncated": False}
            for lv, t in spec]


def handler(provider: str, body: dict):
    user = json.loads(body["messages"][1]["content"])
    if provider == "deepseek":
        return 200, chat_response({"input": user["text"], "spans": ds_spans(user["text"])})
    text = user["input"]
    ds = [{"level": s["level"], "text": s["text"]} for s in sorted(ds_spans(text), key=lambda s: s["start"])]
    ds_slot = "1" if user["phuong_an_1"]["spans"] == ds else "2"
    rule = JUDGE.get(text, "tie")
    winner = {"deepseek": ds_slot, "model": "2" if ds_slot == "1" else "1", "flip": "1"}.get(rule, "tie")
    return 200, chat_response({"winner": winner, "issues_1": [], "issues_2": []}, {"cost": 0.004})


@pytest.fixture()
def pipeline(tmp_path, mock_api):
    mock_api.handler = handler
    p = Pipeline(tmp_path / "p.db", tmp_path / "uploads")
    p.start()
    yield p
    p.stop()


def test_full_flow_decisions_and_report(pipeline, mock_api) -> None:
    texts = list(CASES)
    run_id = pipeline.submit("sample.txt", capture(texts))
    run = pipeline.wait(run_id, timeout=60, poll=0.2)
    decisions = {r.raw_text: r.decision for r in pipeline.store.records(run_id)}
    assert decisions == {
        "Quận 1, Hồ Chí Minh": Decision.AGREE,
        "Bến NghéQuận 1": Decision.DEEPSEEK_BETTER,
        "Tân Phú Hồ Chí Minh": Decision.INCONCLUSIVE,
        "74 Nơ Trang Long": Decision.MODEL_BETTER,
        "Hỏng offset": Decision.DEEPSEEK_INVALID,
    }
    # judge chỉ được gọi cho 3 case khác nhau × 2 thứ tự; DeepSeek gọi lại 1 lần cho case hỏng
    assert len(mock_api.calls_for("qwen")) == 6
    assert len(mock_api.calls_for("deepseek")) == 6
    metrics = run["report"]["metrics"]
    assert metrics["judge_health"]["flip_rate"]["k"] == 1
    assert metrics["model_quality"]["deepseek_better"]["k"] == 1
    assert run["report"]["recommendation"]["status"] == "uncalibrated"
    assert run["cost_usd"] == pytest.approx(metrics["cost"]["cases_total_usd"])
    assert metrics["cost"]["judge_usd"] == pytest.approx(0.024)


def test_judge_input_is_blind_and_canonical(pipeline, mock_api) -> None:
    run_id = pipeline.submit("s.txt", capture(["Bến NghéQuận 1"]))
    pipeline.wait(run_id, timeout=60, poll=0.2)
    for body in mock_api.calls_for("qwen"):
        user = json.loads(body["messages"][1]["content"])
        assert set(user) == {"input", "phuong_an_1", "phuong_an_2"}
        for side in ("phuong_an_1", "phuong_an_2"):
            assert all(set(span) == {"level", "text"} for span in user[side]["spans"])
        assert "deepseek" not in body["messages"][1]["content"].lower()
    orders = [json.loads(b["messages"][1]["content"])["phuong_an_1"] for b in mock_api.calls_for("qwen")]
    assert orders[0] != orders[1]                       # hai lần chấm đảo thứ tự


def test_budget_exceeded_stops_api_calls(pipeline, mock_api, monkeypatch) -> None:
    from feedback.pipeline import runner
    real = runner.load_yaml
    monkeypatch.setattr(runner, "load_yaml",
                        lambda name: {**real(name), "budget": {"per_run_usd": 0.004}} if name == "thresholds" else real(name))
    run_id = pipeline.submit("s.txt", capture(list(CASES)))
    run = pipeline.wait(run_id, timeout=60, poll=0.2)
    decisions = [r.decision for r in pipeline.store.records(run_id)]
    assert Decision.BUDGET_EXCEEDED in decisions
    assert run["cost_usd"] <= 0.004 + 0.003          # vượt tối đa một vòng gọi


def test_fatal_error_marks_failed(pipeline, mock_api) -> None:
    mock_api.handler = lambda p, b: (401, {"error": "bad key"})
    run_id = pipeline.submit("s.txt", capture(["Quận 1, Hồ Chí Minh"]))
    pipeline.wait(run_id, timeout=30, poll=0.2)
    record = pipeline.store.records(run_id)[0]
    assert record.decision is Decision.FAILED
    assert record.layers["deepseek_parse_error"]["type"] == "fatal"


def test_calibration_run_saves_metrics(pipeline, mock_api) -> None:
    texts = ["Bến NghéQuận 1", "74 Nơ Trang Long"]
    gold = {"texts": texts, "results": [
        {"result": {"L4": ["Bến Nghé"], "L3": ["Quận 1"]}},
        {"result": {"L6": ["74"], "L5": ["Nơ Trang Long"]}}]}
    run_id = pipeline.submit("cal.txt", capture(texts), kind="calibration", gold=gold)
    run = pipeline.wait(run_id, timeout=60, poll=0.2)
    cal = run["report"]["calibration_run"]
    assert cal["accuracy"]["k"] == 2 and cal["false_alarm"]["k"] == 0
    assert pipeline.store.latest_calibration(run["meta"]["judge_fingerprint"]) is not None
