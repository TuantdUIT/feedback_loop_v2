"""Chuẩn hoá bản ghi đầu vào về một dạng duy nhất."""

import json
import unicodedata
from pathlib import Path
from typing import Any

from feedback.core.schemas import CaseRecord, ParseResult, Span


METADATA_FIELDS = (
    "trunc",
    "source_id",
    "cut",
    "augment",
    "group",
    "n_anchor",
    "is_full",
    "is_long",
    "len_nfc",
)


def to_nfc(s: str) -> str:
    """Chuẩn hoá chuỗi Unicode về NFC."""
    return unicodedata.normalize("NFC", s)


def extract_text(raw: dict[str, Any]) -> str:
    """Lấy văn bản từ ``text`` hoặc ``input`` và phát hiện xung đột."""
    has_text = "text" in raw
    has_input = "input" in raw
    if has_text and has_input and raw["text"] != raw["input"]:
        raise ValueError("Bản ghi có key 'text' và 'input' khác nhau")
    if has_text:
        return raw["text"]
    if has_input:
        return raw["input"]
    raise ValueError("Bản ghi không có key 'text' hoặc 'input'")


def make_case_id(source_file: str, index: int) -> str:
    """Tạo id từ stem và vị trí; id sẽ đổi nếu thứ tự file nguồn đổi."""
    return f"{Path(source_file).stem}#{index:05d}"


def normalize_record(
    raw: dict[str, Any], source_file: str, index: int
) -> CaseRecord:
    """Chuẩn hoá một dictionary nguồn thành ``CaseRecord``."""
    source_text = extract_text(raw)
    nfc_text = to_nfc(source_text)
    meta = {key: raw[key] for key in METADATA_FIELDS if key in raw}

    nfc_shift = nfc_text != source_text
    if "len_nfc" in raw and len(nfc_text) != raw["len_nfc"]:
        nfc_shift = True
    if nfc_shift:
        meta["nfc_shift"] = True

    old = ParseResult(
        tokens=list(raw["tokens"]),
        bio=list(raw["bio"]),
        spans=[Span.from_dict(span) for span in raw.get("spans", [])],
        source="ml_model",
    )
    return CaseRecord(
        case_id=make_case_id(source_file, index),
        raw_text=nfc_text,
        old=old,
        meta=meta,
    )


def load_records(path: str | Path) -> list[CaseRecord]:
    """Đọc JSON array và chuẩn hoá lần lượt từng bản ghi."""
    source_path = Path(path)
    raw_records = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise ValueError("File đầu vào phải là một JSON array")
    return [
        normalize_record(raw, str(source_path), index)
        for index, raw in enumerate(raw_records)
    ]
