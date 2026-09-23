"""Chuyển output NER dạng list theo level thành CaseRecord của feedback loop."""

from __future__ import annotations

import re
import unicodedata

from feedback.core.schemas import CaseRecord, ParseResult, Span
from kafka_simulation.message import LEVELS


def _overlaps(start: int, end: int, spans: list[Span]) -> bool:
    """Kiểm tra một khoảng có đè lên span đã nhận hay không."""
    return any(start < span.end and end > span.start for span in spans)


def _locate(text: str, entity: str, spans: list[Span]) -> tuple[re.Match[str] | None, bool]:
    """Tìm khớp nguyên văn trước, rồi mới nới khoảng trắng, không fuzzy match."""
    for match in re.finditer(re.escape(entity), text):
        if not _overlaps(match.start(), match.end(), spans):
            return match, False
    parts = entity.split()
    if parts:
        pattern = r"\s*".join(re.escape(part) for part in parts)
        for match in re.finditer(pattern, text):
            if not _overlaps(match.start(), match.end(), spans):
                return match, True
    return None, False


def _tokens_and_bio(text: str, spans: list[Span]) -> tuple[list[str], list[str]]:
    """Cắt tại biên span, khoảng trắng và dấu câu để BIO luôn căn đúng."""
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


def to_case_record(msg: dict) -> CaseRecord:
    """Tạo CaseRecord trung thực; span không định vị được được bỏ và gắn cờ."""
    original = msg["text"]
    text = unicodedata.normalize("NFC", original)
    meta = {
        "api_confidence": msg["confidence"],
        "truncated_unknown": True,
        "source_msg_id": msg["msg_id"],
    }
    if text != original:
        meta["nfc_shift"] = True

    entities = [(level, unicodedata.normalize("NFC", entity))
                for level in LEVELS for entity in msg["result"][level]]
    entities.sort(key=lambda item: -len(item[1]))
    spans: list[Span] = []
    unlocated = []
    loose_matches = []
    for level, entity in entities:
        match, loose = _locate(text, entity, spans) if entity else (None, False)
        if match is None:
            unlocated.append({"level": level, "text": entity})
            continue
        span = Span(start=match.start(), end=match.end(), level=level,
                    text=text[match.start():match.end()], truncated=False)
        spans.append(span)
        if loose:
            loose_matches.append({"level": level, "text": entity})
    spans.sort(key=lambda span: span.start)
    if unlocated:
        meta["unlocated"] = unlocated
    if loose_matches:
        meta["loose_ws_match"] = loose_matches
    tokens, bio = _tokens_and_bio(text, spans)
    return CaseRecord(
        case_id=msg["msg_id"], raw_text=text,
        old=ParseResult(tokens=tokens, bio=bio, spans=spans, source="gsm_api"),
        meta=meta,
    )
