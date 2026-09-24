"""Probe thiên kiến cho judge (BUILD_PIPELINE.md §7.5), dựng từ bộ golden có sẵn output DeepSeek.

Ba loại probe, mỗi probe chấm ở CẢ HAI thứ tự:
  known      (gold, DeepSeek) trên case DeepSeek sai so với gold → gold phải thắng.
  identical  hai bản giống hệt (gold, gold)                     → phải hoà ở cả hai thứ tự (position bias).
  verbose    gold vs gold có một span bị tách vụn thành hai     → gold phải thắng (verbosity bias).
Probe dùng gold làm "sự thật" — với các case mà prompt và gold mâu thuẫn (xem
scripts/external_api/nhuocdiem.md), judge làm đúng prompt vẫn bị tính là sai; đọc kết quả theo từng case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feedback.core.schemas import Span
from kafka_simulation.bridge import to_case_record


@dataclass
class Probe:
    kind: str
    text: str
    good: list[Span]          # phương án đúng (gold)
    other: list[Span]         # phương án kia
    expected: str             # "good" | "tie"
    source: str


def gold_spans(text: str, result: dict[str, Any]) -> list[Span]:
    msg = {"msg_id": "probe#00000", "text": text,
           "result": {f"L{i}": list(result.get(f"L{i}", [])) for i in range(1, 8)}}
    return to_case_record(msg).old.spans


def split_longest(spans: list[Span]) -> list[Span] | None:
    """Tách span nhiều từ dài nhất thành hai span cùng level (tách vụn)."""
    multi = [s for s in spans if " " in s.text.strip()]
    if not multi:
        return None
    target = max(multi, key=lambda s: len(s.text))
    cut = target.text.index(" ")
    first = Span(target.start, target.start + cut, target.level, target.text[:cut])
    rest = target.text[cut + 1:]
    second = Span(target.start + cut + 1, target.end, target.level, rest)
    return sorted([s for s in spans if s != target] + [first, second])


def build(test_dir: Path, golden_json: Path, n_identical: int = 5, n_verbose: int = 5) -> list[Probe]:
    golden = json.loads(golden_json.read_text(encoding="utf-8"))
    gold_by_text = {t: r["result"] for t, r in zip(golden["texts"], golden["results"], strict=True)}
    raw = {r["text"]: r for r in map(json.loads, (test_dir / "raw_deepseek.jsonl").read_text(encoding="utf-8").splitlines())}
    preds = [json.loads(line) for line in (test_dir / "pred_deepseek.jsonl").read_text(encoding="utf-8").splitlines()]

    probes: list[Probe] = []
    correct = []
    for pred in preds:
        text = pred["text"]
        good = gold_spans(text, gold_by_text[text])
        ds = [Span(s["start"], s["end"], s["level"], s["text"]) for s in raw[text]["parsed"]["spans"]]
        if pred["exact"]:
            correct.append((text, good))
        else:
            probes.append(Probe("known", text, good, ds, "good", f"{test_dir.name}#{pred['i']}"))
    picked = [c for c in correct if len(c[1]) >= 2]
    for text, good in picked[:n_identical]:
        probes.append(Probe("identical", text, good, list(good), "tie", test_dir.name))
    verbose = [(t, g, split_longest(g)) for t, g in picked[n_identical:]]
    for text, good, split in [v for v in verbose if v[2]][:n_verbose]:
        probes.append(Probe("verbose", text, good, split, "good", test_dir.name))
    return probes


def score(probe: Probe, winners: list[str]) -> dict[str, Any]:
    """``winners`` là bên thắng theo từng thứ tự, đã quy về 'good' | 'other' | 'tie'."""
    consistent = winners[0] == winners[1]
    verdict = winners[0] if consistent else None
    return {"kind": probe.kind, "text": probe.text, "expected": probe.expected, "winners": winners,
            "consistent": consistent, "correct": verdict == probe.expected}
