"""Giai đoạn ``qwen_judge``: Qwen chấm model vs DeepSeek, hai lần với thứ tự đảo ngược.

Chống thiên kiến (BUILD_PIPELINE.md §6.2):
  position         chấm (model, DeepSeek) rồi (DeepSeek, model); chỉ nhận khi hai lần cùng chỉ một bên.
  verbosity        hai bản về cùng dạng chuẩn: chỉ ``level`` + ``text`` theo vị trí; rubric phạt span thừa.
  self-enhancement Qwen không parse ở đâu trong pipeline; nguồn gốc bị ẩn thành "phương án 1/2".
Bỏ ``start``/``end``/``truncated``/``tokens``/``bio`` khỏi input của judge: model đang triển khai không
có ``truncated`` còn DeepSeek có — giữ lại sẽ để lộ nguồn và làm bản DeepSeek trông "đầy đủ hơn".
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from feedback.core import llm_client
from feedback.core.config import load_yaml
from feedback.core.prompt_loader import load_prompt, load_task_prompt
from feedback.core.schemas import CaseRecord, Span

ORDERS = (("model", "deepseek"), ("deepseek", "model"))
_WINNER_ALIASES = {"1": "1", "2": "2", "tie": "tie", "hoa": "tie", "hoà": "tie", "hòa": "tie"}


def canonical(spans: list[Span]) -> dict[str, Any]:
    """Dạng chuẩn đưa cho judge: chỉ level + text, theo thứ tự xuất hiện."""
    return {"spans": [{"level": s.level, "text": s.text} for s in sorted(spans, key=lambda s: (s.start, s.end))]}


def normalize_winner(parsed: Any) -> str | None:
    if not isinstance(parsed, dict):
        return None
    return _WINNER_ALIASES.get(str(parsed.get("winner", "")).strip().lower())


def system_prompt() -> tuple[str, dict[str, str]]:
    core, core_sha = load_prompt("v2_with_partial_input")
    task, task_sha = load_task_prompt("judge")
    return core + "\n\n" + task, {"prompt_sha": core_sha, "judge_sha": task_sha}


def fingerprint() -> tuple[str, dict[str, Any]]:
    """Dấu vân tay cấu hình judge: kết quả hiệu chuẩn chỉ đúng cho đúng cấu hình này."""
    _, shas = system_prompt()
    cfg = load_yaml("models")["qwen"]
    parts = {**shas, "model": llm_client.provider_config("qwen")["model"],
             "reasoning_budgets": cfg.get("reasoning_budgets"),
             **{k: cfg.get(k) for k in ("temperature", "presence_penalty", "top_p", "top_k")}}
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return digest, parts


def judge_pair(text: str, first: list[Span], second: list[Span], budget: llm_client.Budget | None,
               system: str, overrides: dict[str, Any] | None = None) -> tuple[str, llm_client.LLMCall]:
    """Một lần chấm; trả ('1' | '2' | 'tie', call)."""
    user = json.dumps({"input": text, "phuong_an_1": canonical(first), "phuong_an_2": canonical(second)},
                      ensure_ascii=False)
    call = llm_client.chat("qwen", system, user, budget=budget, overrides=overrides,
                           accept=lambda parsed: normalize_winner(parsed) is not None)
    return normalize_winner(call.parsed), call


def run(record: CaseRecord, budget: llm_client.Budget | None) -> tuple[CaseRecord, str]:
    system, shas = system_prompt()
    sides = {"model": record.old.spans, "deepseek": record.new.spans}
    calls: list[dict[str, Any]] = []
    for order in ORDERS:
        try:
            raw, call = judge_pair(record.raw_text, sides[order[0]], sides[order[1]], budget, system)
        except llm_client.LLMError as exc:
            record.cost_usd += exc.cost_usd
            exc.cost_usd = 0.0
            raise
        record.cost_usd += call.cost_usd
        winner = "tie" if raw == "tie" else order[int(raw) - 1]
        issues = {order[0]: call.parsed.get("issues_1") or [], order[1]: call.parsed.get("issues_2") or []}
        calls.append({"order": list(order), "raw_winner": raw, "winner": winner, "issues": issues,
                      "cost_usd": call.cost_usd, "usage": call.usage, "elapsed_sec": call.elapsed_sec,
                      "attempts": call.attempts})

    winners = [c["winner"] for c in calls]
    consistent = winners[0] == winners[1]
    record.layers["judge"] = {**shas, "model": llm_client.provider_config("qwen")["model"], "calls": calls,
                              "winners": winners, "consistent": consistent,
                              "verdict": winners[0] if consistent else None}
    return record, "judged"
