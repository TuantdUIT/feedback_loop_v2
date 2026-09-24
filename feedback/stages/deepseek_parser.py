"""Giai đoạn ``deepseek_parse``: DeepSeek parse lại input bằng prompt lõi v2, rồi qua Lớp 0.

Bản parse của DeepSeek là GIẢ THUYẾT CẠNH TRANH để judge so với model đang triển khai, không phải
đáp án đúng (trên golden_test2 DeepSeek chỉ đạt 72% accuracy ở nhóm gõ dở). Vi phạm hard ở Lớp 0
→ gọi lại ``thresholds.deepseek.retry_on_hard_violation`` lần; vẫn sai → ``deepseek_invalid``.
"""

from __future__ import annotations

import json
from typing import Any

from feedback.core import llm_client
from feedback.core.bio import tokens_and_bio
from feedback.core.prompt_loader import load_prompt
from feedback.core.schemas import CaseRecord, Decision, ParseResult, Span
from feedback.stages.validator import validate_parse


def _record_of(parsed: Any) -> dict[str, Any] | None:
    """Prompt cho một input trả một object; chịu được mảng một phần tử hoặc bọc trong ``records``."""
    if isinstance(parsed, list) and len(parsed) == 1:
        parsed = parsed[0]
    if isinstance(parsed, dict) and isinstance(parsed.get("records"), list) and len(parsed["records"]) == 1:
        parsed = parsed["records"][0]
    return parsed if isinstance(parsed, dict) and isinstance(parsed.get("spans"), list) else None


def _spans(raw: list[Any]) -> tuple[list[Span], list[dict[str, Any]]]:
    spans, broken = [], []
    for item in raw:
        try:
            spans.append(Span(start=int(item["start"]), end=int(item["end"]), level=str(item["level"]),
                              text=str(item["text"]), truncated=bool(item.get("truncated", False))))
        except (KeyError, TypeError, ValueError):
            broken.append({"rule": "span_malformed", "item": item})
    return spans, broken


def parse(record: CaseRecord, budget: llm_client.Budget | None, retry_on_hard: int = 1) -> tuple[CaseRecord, str]:
    """Gắn ``record.new`` và trace; trả (record, topic kế tiếp)."""
    system, prompt_sha = load_prompt("v2_with_partial_input")
    user = json.dumps({"text": record.raw_text}, ensure_ascii=False)
    calls: list[dict[str, Any]] = []
    layer0: dict[str, Any] = {}

    for round_ in range(retry_on_hard + 1):
        try:
            call = llm_client.chat("deepseek", system, user, budget=budget,
                                   accept=lambda parsed: _record_of(parsed) is not None)
        except llm_client.LLMError as exc:
            record.cost_usd += exc.cost_usd
            exc.cost_usd = 0.0                      # đã ghi vào record, tránh cộng lần nữa ở worker
            record.layers.setdefault("deepseek", {"calls": calls, "prompt_sha": prompt_sha})
            raise
        record.cost_usd += call.cost_usd
        calls.append({"round": round_, "model": call.model, "cost_usd": call.cost_usd, "usage": call.usage,
                      "elapsed_sec": call.elapsed_sec, "attempts": call.attempts})
        parsed = _record_of(call.parsed)
        spans, broken = _spans(parsed["spans"])
        tokens, bio = parsed.get("tokens"), parsed.get("bio")
        if not (isinstance(tokens, list) and isinstance(bio, list)):
            tokens = bio = None
        layer0 = validate_parse(record.raw_text, spans, tokens, bio)
        layer0["hard"] = broken + layer0["hard"]
        layer0["ok"] = not layer0["hard"]
        if layer0["ok"]:
            break

    fixed = layer0.pop("spans")
    record.layers["deepseek"] = {"calls": calls, "prompt_sha": prompt_sha,
                                 "layer0": {k: layer0[k] for k in ("ok", "hard", "soft")}}
    if not layer0["ok"]:
        record.decision = Decision.DEEPSEEK_INVALID
        return record, "decided"
    canonical_tokens, canonical_bio = tokens_and_bio(record.raw_text, fixed)
    record.new = ParseResult(tokens=canonical_tokens, bio=canonical_bio, spans=fixed, source="deepseek")
    return record, "parsed"
