"""Test cho core.llm_client qua mock HTTP: retry, lỗi fatal, response_format, cost, ngân sách."""

import time

import pytest

from feedback.core import llm_client
from feedback.core.llm_client import (BudgetExceeded, FatalAPIError, TransientAPIError, chat,
                                      deepseek_cost, provider_config)
from feedback.tests.conftest import chat_response

NO_SLEEP = lambda seconds: None  # noqa: E731


class FakeBudget:
    def __init__(self, limit: float) -> None:
        self.limit, self.spent, self.reserved = limit, 0.0, 0.0

    def reserve(self, estimate: float) -> bool:
        if self.spent + self.reserved + estimate > self.limit:
            return False
        self.reserved += estimate
        return True

    def settle(self, estimate: float, actual: float) -> None:
        self.reserved -= estimate
        self.spent += actual


def test_deepseek_success_and_cost(mock_api) -> None:
    mock_api.handler = lambda p, b: (200, chat_response({"spans": []}))
    call = chat("deepseek", "sys", "user", sleep=NO_SLEEP)
    assert call.parsed == {"spans": []}
    body = mock_api.calls_for("deepseek")[0]
    assert body["response_format"] == {"type": "json_object"} and body["max_tokens"] == 12000
    # 100 miss × 0.15 + 900 hit × 0.003 + 50 out × 0.6, thấp hoặc cao điểm
    base = (100 * 0.15 + 900 * 0.003 + 50 * 0.6) / 1e6
    assert call.cost_usd in (pytest.approx(base), pytest.approx(2 * base))


def test_peak_hours_double_price(mock_api) -> None:
    cfg = provider_config("deepseek")
    usage = {"prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 1_000_000, "completion_tokens": 0}
    wed_03 = time.strptime("2026-09-23T03:00:00", "%Y-%m-%dT%H:%M:%S")
    wed_15 = time.strptime("2026-09-23T15:00:00", "%Y-%m-%dT%H:%M:%S")
    sat_03 = time.strptime("2026-09-26T03:00:00", "%Y-%m-%dT%H:%M:%S")
    assert deepseek_cost(cfg, usage, wed_03) == pytest.approx(0.30)
    assert deepseek_cost(cfg, usage, wed_15) == pytest.approx(0.15)
    assert deepseek_cost(cfg, usage, sat_03) == pytest.approx(0.15)


def test_empty_output_retried_and_cost_accumulates(mock_api) -> None:
    replies = iter([chat_response(""), chat_response({"winner": "1"})])
    mock_api.handler = lambda p, b: (200, next(replies))
    call = chat("deepseek", "s", "u", sleep=NO_SLEEP)
    assert call.parsed == {"winner": "1"} and len(call.attempts) == 2 and call.attempts[0]["empty"]
    assert call.cost_usd > 0


def test_qwen_reasoning_budget_ladder_and_openrouter_cost(mock_api) -> None:
    replies = iter([chat_response("", {"cost": 0.01}), chat_response("", {"cost": 0.02}),
                    chat_response({"winner": "tie"}, {"cost": 0.005})])
    mock_api.handler = lambda p, b: (200, next(replies))
    call = chat("qwen", "s", "u", sleep=NO_SLEEP)
    budgets = [b["reasoning"]["max_tokens"] for b in mock_api.calls_for("qwen")]
    assert budgets == [10000, 5000, 3000]
    assert call.cost_usd == pytest.approx(0.035)


def test_fatal_http_stops_immediately(mock_api) -> None:
    mock_api.handler = lambda p, b: (402, {"error": {"message": "Insufficient Balance"}})
    with pytest.raises(FatalAPIError):
        chat("deepseek", "s", "u", sleep=NO_SLEEP)
    assert len(mock_api.calls_for("deepseek")) == 1


def test_transient_exhausts_attempts(mock_api) -> None:
    mock_api.handler = lambda p, b: (503, {"error": "busy"})
    with pytest.raises(TransientAPIError):
        chat("deepseek", "s", "u", sleep=NO_SLEEP)
    assert len(mock_api.calls_for("deepseek")) == 3


def test_response_format_disabled_when_rejected(mock_api) -> None:
    def handler(p, body):
        if "response_format" in body:
            return 400, {"error": {"message": "response_format not supported"}}
        return 200, chat_response({"spans": []})
    mock_api.handler = handler
    call = chat("deepseek", "s", "u", sleep=NO_SLEEP)
    assert call.parsed == {"spans": []}
    assert llm_client._RESPONSE_FORMAT_OK["deepseek"] is False


def test_budget_blocks_before_sending(mock_api) -> None:
    mock_api.handler = lambda p, b: (200, chat_response({"spans": []}))
    with pytest.raises(BudgetExceeded):
        chat("deepseek", "s", "u", budget=FakeBudget(limit=0.001), sleep=NO_SLEEP)
    assert mock_api.calls_for("deepseek") == []


def test_budget_settles_actual_cost(mock_api) -> None:
    mock_api.handler = lambda p, b: (200, chat_response({"spans": []}))
    budget = FakeBudget(limit=1.0)
    call = chat("deepseek", "s", "u", budget=budget, sleep=NO_SLEEP)
    assert budget.reserved == pytest.approx(0) and budget.spent == pytest.approx(call.cost_usd)
