"""Lớp 0 — kiểm tra hình thức bằng Python thuần, không gọi LLM.

Vi phạm chia hai mức:
  hard  chắc chắn sai ở SPAN: level ngoài L1–L7, offset ngoài chuỗi, ``text`` không có trong input,
        span rỗng, span chồng lấn. Pipeline chỉ dùng span để so sánh và chấm, nên đây là lỗi chặn.
  soft  đáng ngờ nhưng không chặn: offset lệch đã tự sửa được, tokens/BIO không khớp span, tag
        ngoài hệ L, transition IOB2 sai. Tokens/BIO sinh lại được từ span, nên gọi lại LLM chỉ vì
        BIO sai là tốn tiền vô ích.

Offset lệch được sửa khi ``text`` của span xuất hiện đúng MỘT chỗ còn trống trong input; xuất hiện
nhiều chỗ thì không đoán, coi là hard. Không đánh giá đúng sai ngữ nghĩa.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from feedback.core.bio import LEVELS, invalid_tags, spans_from_bio, transition_errors
from feedback.core.schemas import Span


def _free_occurrences(text: str, needle: str, taken: list[Span]) -> list[int]:
    found, start = [], text.find(needle)
    while start >= 0:
        end = start + len(needle)
        if not any(start < s.end and end > s.start for s in taken):
            found.append(start)
        start = text.find(needle, start + 1)
    return found


def validate_spans(text: str, spans: list[Span]) -> tuple[list[Span], list[dict[str, Any]], list[dict[str, Any]]]:
    """Trả (span đã sửa offset, vi phạm hard, vi phạm soft)."""
    hard: list[dict[str, Any]] = []
    soft: list[dict[str, Any]] = []
    good: list[Span] = []
    pending: list[Span] = []
    for span in spans:
        where = {"level": span.level, "text": span.text, "start": span.start, "end": span.end}
        if span.level not in LEVELS:
            hard.append({"rule": "level_invalid", **where})
        elif not span.text:
            hard.append({"rule": "span_empty", **where})
        elif 0 <= span.start < span.end <= len(text) and text[span.start:span.end] == span.text:
            good.append(span)
        else:
            pending.append(span)

    for span in pending:
        where = {"level": span.level, "text": span.text, "start": span.start, "end": span.end}
        places = _free_occurrences(text, span.text, good)
        if len(places) == 1:
            fixed = replace(span, start=places[0], end=places[0] + len(span.text))
            good.append(fixed)
            soft.append({"rule": "offset_repaired", **where, "fixed_start": fixed.start})
        elif places:
            hard.append({"rule": "offset_ambiguous", **where, "candidates": places})
        else:
            hard.append({"rule": "text_not_in_input", **where})

    good.sort()
    for a, b in zip(good, good[1:]):
        if b.start < a.end:
            hard.append({"rule": "overlap", "a": a.text, "b": b.text, "start": b.start})
    return good, hard, soft


def validate_bio(text: str, tokens: list[str], bio: list[str], spans: list[Span]) -> list[dict[str, Any]]:
    """Vi phạm soft của tokens/BIO so với chính span của bản parse."""
    soft: list[dict[str, Any]] = []
    if len(tokens) != len(bio):
        return [{"rule": "tokens_bio_length", "tokens": len(tokens), "bio": len(bio)}]
    if bad := invalid_tags(bio):
        soft.append({"rule": "tag_invalid", "positions": bad})
    if bad := transition_errors(bio):
        soft.append({"rule": "iob2_transition", "positions": bad})
    derived = spans_from_bio(text, tokens, bio)
    if derived is None:
        soft.append({"rule": "tokens_not_in_input"})
    elif [(s.start, s.end, s.level) for s in derived] != [(s.start, s.end, s.level) for s in sorted(spans)]:
        soft.append({"rule": "bio_spans_mismatch"})
    return soft


def validate_parse(text: str, spans: list[Span], tokens: list[str] | None = None,
                   bio: list[str] | None = None) -> dict[str, Any]:
    """Kiểm tra một bản parse; ``spans`` trả về đã được sửa offset nếu sửa được."""
    fixed, hard, soft = validate_spans(text, spans)
    if tokens is not None and bio is not None:
        soft += validate_bio(text, tokens, bio, fixed)
    return {"ok": not hard, "hard": hard, "soft": soft, "spans": fixed}
