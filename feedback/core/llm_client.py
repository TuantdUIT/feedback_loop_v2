"""Lớp bọc chung để gọi LLM qua API: DeepSeek (parser) và Qwen qua OpenRouter (judge).

Trách nhiệm: dựng request, retry trong một lần gọi, ép output JSON, tính token + chi phí, và giữ
chỗ ngân sách trước MỖI lần gửi request. Không chứa prompt nhiệm vụ hay logic gate.

Chống kẹt vòng lặp suy luận (đo trên golden, xem scripts/external_api/*):
  - ``max_tokens`` là chốt chặn cuối;
  - Qwen: ``reasoning.max_tokens`` giảm dần qua từng lần thử (``reasoning_budgets``);
  - ``response_format=json_object``, tự tắt nếu provider từ chối;
  - retry khi trả về RỖNG / không đọc được JSON dù đã tốn phí.

Lỗi được chia để worker xử lý đúng:
  FatalAPIError      sai key, hết tiền, sai tham số — gọi lại vô ích.
  TransientAPIError  hết lượt thử vì lỗi mạng/429/5xx/rỗng — broker có thể retry cả message.
  BudgetExceeded     giữ chỗ ngân sách thất bại — không gửi request.
Cả ba mang ``cost_usd`` đã tiêu trước khi lỗi, để chi phí không bị mất khỏi sổ sách.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from feedback.core.config import env, load_yaml


class Budget(Protocol):
    def reserve(self, estimate_usd: float) -> bool: ...
    def settle(self, estimate_usd: float, actual_usd: float) -> None: ...


class LLMError(Exception):
    def __init__(self, message: str, cost_usd: float = 0.0, attempts: list | None = None) -> None:
        super().__init__(message)
        self.cost_usd = cost_usd
        self.attempts = attempts or []


class FatalAPIError(LLMError):
    pass


class TransientAPIError(LLMError):
    pass


class BudgetExceeded(LLMError):
    pass


@dataclass
class LLMCall:
    content: str
    parsed: Any
    model: str
    cost_usd: float
    usage: dict[str, int]
    elapsed_sec: float
    attempts: list[dict[str, Any]] = field(default_factory=list)


_RESPONSE_FORMAT_OK: dict[str, bool] = {}


def extract_json(text: str | None) -> Any:
    """Đọc JSON từ nội dung trả về; chịu được code fence và văn bản thừa quanh JSON."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    raw = (fence.group(1) if fence else text).strip()
    try:
        return json.loads(raw)
    except ValueError:
        block = re.search(r"[\{\[].*[\}\]]", raw, re.S)
        if block:
            try:
                return json.loads(block.group(0))
            except ValueError:
                return None
    return None


def provider_config(provider: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**load_yaml("models")[provider], **(overrides or {})}
    cfg["model"] = env(cfg["model_env"])
    cfg["api_key"] = env(cfg["key_env"])
    cfg["url"] = (env(cfg.get("base_url_env", "")) or cfg["base_url"]).rstrip("/") + "/chat/completions"
    if not cfg["model"] or not cfg["api_key"]:
        raise FatalAPIError(f"Thiếu {cfg['model_env']} hoặc {cfg['key_env']} trong .env")
    return cfg


def _is_peak(utc: time.struct_time, hours: list[list[int]]) -> bool:
    return utc.tm_wday < 5 and any(a <= utc.tm_hour < b for a, b in hours)


def deepseek_cost(cfg: dict[str, Any], usage: dict[str, Any], sent: time.struct_time) -> float:
    """Chi phí từ token × bảng giá; API DeepSeek không trả chi phí."""
    price = cfg["price_per_1m"].get(cfg["model"])
    if price is None:
        raise FatalAPIError(f"Chưa có đơn giá cho model {cfg['model']!r} trong models.yaml")
    hit = usage.get("prompt_cache_hit_tokens") or 0
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is None:
        miss = (usage.get("prompt_tokens") or 0) - hit
    out = usage.get("completion_tokens") or 0
    mult = cfg["peak_multiplier"] if _is_peak(sent, cfg["peak_hours_utc"]) else 1
    return mult * (miss * price["miss"] + hit * price["hit"] + out * price["out"]) / 1e6


def _body(provider: str, cfg: dict[str, Any], system: str, user: str, attempt: int) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if provider == "qwen":
        budgets = cfg["reasoning_budgets"]
        body["reasoning"] = {"max_tokens": budgets[min(attempt, len(budgets) - 1)]}
        body["usage"] = {"include": True}
        body["temperature"] = cfg["temperature"]
        for key in ("presence_penalty", "top_p", "top_k"):
            if cfg.get(key) is not None:
                body[key] = cfg[key]
    else:
        body["temperature"] = 0            # chế độ suy luận của DeepSeek bỏ qua tham số này
    if _RESPONSE_FORMAT_OK.get(provider, True):
        body["response_format"] = {"type": "json_object"}
    return body


def _post(cfg: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        cfg["url"], data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
        headers={"Authorization": "Bearer " + cfg["api_key"], "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=cfg["timeout_sec"]) as response:
        return json.loads(response.read())


def chat(provider: str, system: str, user: str, *, budget: Budget | None = None,
         accept: Callable[[Any], bool] = lambda parsed: parsed is not None,
         sleep: Callable[[float], None] = time.sleep, overrides: dict[str, Any] | None = None) -> LLMCall:
    """Gọi một provider tới khi có output ``accept`` được, hoặc hết lượt thử.

    ``overrides`` ghi đè cấu hình của ``models.yaml`` cho riêng lần gọi này (dùng khi đo tham số).
    """
    cfg = provider_config(provider, overrides)
    n_attempts = len(cfg["reasoning_budgets"]) if provider == "qwen" else cfg["max_attempts"]
    estimate = float(cfg["estimate_usd_per_call"])
    total_cost = 0.0
    tokens = {"prompt": 0, "cache_hit": 0, "completion": 0, "reasoning": 0}
    attempts: list[dict[str, Any]] = []
    started = time.monotonic()

    attempt = 0
    while attempt < n_attempts:
        if budget is not None and not budget.reserve(estimate):
            raise BudgetExceeded("Chạm trần chi phí của run", total_cost, attempts)
        sent = time.gmtime()
        record: dict[str, Any] = {"attempt": attempt, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", sent)}
        if provider == "qwen":
            record["reasoning_budget"] = _body(provider, cfg, "", "", attempt)["reasoning"]["max_tokens"]
        cost = 0.0
        try:
            response = _post(cfg, _body(provider, cfg, system, user, attempt))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            if budget is not None:
                budget.settle(estimate, 0.0)
            if exc.code == 400 and _RESPONSE_FORMAT_OK.get(provider, True) and "response_format" in detail:
                _RESPONSE_FORMAT_OK[provider] = False
                attempts.append({**record, "note": "tắt response_format"})
                continue                                   # không tính là một lượt thử
            attempts.append({**record, "http_error": exc.code, "detail": detail[:200]})
            fatal = exc.code in (401, 402, 422) or (exc.code == 403 and "limit" in detail.lower())
            if fatal:
                raise FatalAPIError(f"HTTP {exc.code}: {detail[:150]}", total_cost, attempts) from exc
            attempt += 1
            if attempt < n_attempts:
                sleep((10 if exc.code in (429, 503) else 4) * attempt)
            continue
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            if budget is not None:
                budget.settle(estimate, 0.0)
            attempts.append({**record, "error": str(exc)[:200]})
            attempt += 1
            if attempt < n_attempts:
                sleep(4 * attempt)
            continue

        usage = response.get("usage") or {}
        cost = float(usage.get("cost") or 0) if provider == "qwen" else deepseek_cost(cfg, usage, sent)
        total_cost += cost
        if budget is not None:
            budget.settle(estimate, cost)
        details = usage.get("completion_tokens_details") or {}
        tokens["prompt"] += usage.get("prompt_tokens") or 0
        tokens["cache_hit"] += usage.get("prompt_cache_hit_tokens") or (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        tokens["completion"] += usage.get("completion_tokens") or 0
        tokens["reasoning"] += details.get("reasoning_tokens") or 0
        choice = (response.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        parsed = extract_json(content)
        ok = accept(parsed)
        attempts.append({**record, "cost_usd": round(cost, 8), "finish_reason": choice.get("finish_reason"),
                         "completion_tokens": usage.get("completion_tokens"),
                         "reasoning_tokens": details.get("reasoning_tokens"), "empty": not ok})
        if ok:
            return LLMCall(content=content, parsed=parsed, model=cfg["model"], cost_usd=total_cost,
                           usage=tokens, elapsed_sec=round(time.monotonic() - started, 2), attempts=attempts)
        attempt += 1
        if attempt < n_attempts:
            sleep(2)

    raise TransientAPIError(f"{provider}: hết {n_attempts} lượt thử không có output hợp lệ",
                            total_cost, attempts)
