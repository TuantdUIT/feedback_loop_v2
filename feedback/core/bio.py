"""Chuyển đổi hai chiều giữa chuỗi BIO và danh sách span, và kiểm tra transition IOB2.

Chỉ chấp nhận tag hệ L (``O``, ``B-L1``…``B-L7``, ``I-L1``…``I-L7``). ``I-Lx`` sau ``O`` hoặc sau
``B-Ly``/``I-Ly`` khác loại là không hợp lệ. Module không gọi LLM và không tự sửa nhãn ngữ nghĩa.
"""

from __future__ import annotations

import re

from feedback.core.schemas import Span


LEVELS = tuple(f"L{i}" for i in range(1, 8))
VALID_TAGS = frozenset({"O"} | {f"{p}-{lv}" for p in "BI" for lv in LEVELS})


def invalid_tags(bio: list[str]) -> list[int]:
    """Vị trí các tag không thuộc hệ L."""
    return [i for i, tag in enumerate(bio) if tag not in VALID_TAGS]


def transition_errors(bio: list[str]) -> list[int]:
    """Vị trí các ``I-X`` không đứng sau ``B-X``/``I-X`` cùng loại."""
    errors = []
    previous = "O"
    for i, tag in enumerate(bio):
        if tag.startswith("I-") and (previous == "O" or previous[2:] != tag[2:]):
            errors.append(i)
        previous = tag
    return errors


def locate_tokens(text: str, tokens: list[str]) -> list[tuple[int, int]] | None:
    """Offset của từng token theo thứ tự; None nếu token không khớp văn bản hoặc bỏ sót ký tự.

    Giữa hai token liên tiếp chỉ được có khoảng trắng — nếu có ký tự khác nghĩa là tokens đã bỏ
    mất một phần văn bản.
    """
    offsets = []
    position = 0
    for token in tokens:
        if not token:
            return None
        found = text.find(token, position)
        if found < 0 or text[position:found].strip():
            return None
        offsets.append((found, found + len(token)))
        position = found + len(token)
    if text[position:].strip():
        return None
    return offsets


def spans_from_bio(text: str, tokens: list[str], bio: list[str]) -> list[Span] | None:
    """Gộp các chuỗi B/I liền kề thành span; None nếu không định vị được token."""
    offsets = locate_tokens(text, tokens)
    if offsets is None or len(offsets) != len(bio):
        return None
    spans: list[Span] = []
    start = level = None
    end = 0
    for (tok_start, tok_end), tag in zip(offsets, bio):
        if tag.startswith("I-") and level == tag[2:]:
            end = tok_end
            continue
        if level is not None:
            spans.append(Span(start=start, end=end, level=level, text=text[start:end]))
            level = None
        if tag.startswith(("B-", "I-")):
            start, end, level = tok_start, tok_end, tag[2:]
    if level is not None:
        spans.append(Span(start=start, end=end, level=level, text=text[start:end]))
    return spans


def tokens_and_bio(text: str, spans: list[Span]) -> tuple[list[str], list[str]]:
    """Cắt tại biên span, khoảng trắng và dấu câu để BIO luôn căn đúng với span."""
    cuts = {0, len(text)}
    for span in spans:
        cuts.update((span.start, span.end))
    for match in re.finditer(r"\s+", text):
        cuts.update((match.start(), match.end()))
    for index, char in enumerate(text):
        if not char.isalnum() and not char.isspace():
            cuts.update((index, index + 1))

    tokens: list[str] = []
    bio: list[str] = []
    points = sorted(cuts)
    for start, end in zip(points, points[1:]):
        token = text[start:end]
        if not token or token.isspace():
            continue
        owner = next((span for span in spans if span.start <= start and end <= span.end), None)
        label = "O" if owner is None else ("B-" if start == owner.start else "I-") + owner.level
        tokens.append(token)
        bio.append(label)
    return tokens, bio
