"""Test chuẩn hoá dữ liệu thật và round-trip schema."""

import json
import unicodedata
from pathlib import Path

import pytest

from feedback.core.normalize import (
    METADATA_FIELDS,
    extract_text,
    load_records,
    normalize_record,
    to_nfc,
)
from feedback.core.schemas import CaseRecord, Decision, ParseResult, Span


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_MODEL = PROJECT_ROOT / "output_model.json"


@pytest.fixture(scope="module")
def raw_records() -> list[dict]:
    """Đọc ba bản ghi thật đầu tiên, không dựng fixture giả."""
    return json.loads(OUTPUT_MODEL.read_text(encoding="utf-8"))[:3]


def test_nfc_idempotent() -> None:
    text = "Hô\u0300 Chí Minh"
    assert to_nfc(to_nfc(text)) == to_nfc(text)


def test_nfc_vietnamese_nfd() -> None:
    text = "Hô\u0300 Chí Minh"
    normalized = to_nfc(text)
    assert normalized == "Hồ Chí Minh"
    assert len(normalized) < len(text)
    assert unicodedata.is_normalized("NFC", normalized)


def test_nfc_shift_flagged() -> None:
    text = "Hô\u0300 Chí Minh, Quận 1"
    raw = {
        "text": text,
        "tokens": ["Hô\u0300", "Chí", "Minh", ",", "Quận", "1"],
        "bio": ["B-L2", "I-L2", "I-L2", "O", "B-L3", "I-L3"],
        "spans": [
            {
                "level": "L2",
                "start": 0,
                "end": 12,
                "text": "Hô\u0300 Chí Minh",
                "truncated": False,
            }
        ],
        "len_nfc": len(to_nfc(text)),
    }
    record = normalize_record(raw, "nfd.json", 0)
    assert record.meta["nfc_shift"] is True
    assert record.old.spans[0].start == 0
    assert record.old.spans[0].end == 12


def test_key_text(raw_records: list[dict]) -> None:
    record = normalize_record(raw_records[0], "output_model.json", 0)
    assert record.raw_text == raw_records[0]["text"]
    assert record.old.source == "ml_model"


def test_key_input() -> None:
    raw = {"input": "ABC", "tokens": ["ABC"], "bio": ["O"], "spans": []}
    record = normalize_record(raw, "result.json", 0)
    assert record.raw_text == "ABC"


def test_key_conflict() -> None:
    with pytest.raises(ValueError, match="khác nhau"):
        extract_text({"text": "A", "input": "B"})


def test_metadata_preserved(raw_records: list[dict]) -> None:
    raw = raw_records[0]
    record = normalize_record(raw, "output_model.json", 0)
    expected = {key: raw[key] for key in METADATA_FIELDS}
    assert record.meta == expected


def test_metadata_absent() -> None:
    raw = {"input": "ABC", "tokens": ["ABC"], "bio": ["O"], "spans": []}
    assert normalize_record(raw, "result.json", 0).meta == {}


def test_span_offsets_match(raw_records: list[dict]) -> None:
    for index, raw in enumerate(raw_records):
        record = normalize_record(raw, "output_model.json", index)
        for span in record.old.spans:
            assert span.text == record.raw_text[span.start : span.end]


def test_truncated_preserved(raw_records: list[dict]) -> None:
    record = normalize_record(raw_records[0], "output_model.json", 0)
    assert [span.truncated for span in record.old.spans] == [False, False, True]


def test_empty_spans() -> None:
    raw = {"input": "ABC", "tokens": ["ABC"], "bio": ["O"], "spans": []}
    assert normalize_record(raw, "result.json", 0).old.spans == []


def test_bio_length(raw_records: list[dict]) -> None:
    for index, raw in enumerate(raw_records):
        record = normalize_record(raw, "output_model.json", index)
        assert len(record.old.bio) == len(record.old.tokens)


def test_roundtrip(raw_records: list[dict]) -> None:
    record = normalize_record(raw_records[0], "output_model.json", 0)
    record.new = ParseResult.from_dict(record.old.to_dict())
    record.new.source = "slm_repair"
    record.attach("l0", {"hard": [], "soft": []})
    record.decision = Decision.ACCEPT_NEW
    restored = CaseRecord.from_dict(record.to_dict())
    assert restored == record
    assert restored.old.spans[0].to_dict() == {
        "level": "L6",
        "start": 0,
        "end": 3,
        "text": "444",
        "truncated": False,
    }


def test_attach_is_append_only(raw_records: list[dict]) -> None:
    record = normalize_record(raw_records[0], "output_model.json", 0)
    record.attach("l0", {"hard": []})
    with pytest.raises(ValueError, match="đã có dữ liệu"):
        record.attach("l0", {"hard": ["offset"]})


def test_case_id_stable() -> None:
    first = load_records(OUTPUT_MODEL)
    second = load_records(OUTPUT_MODEL)
    assert [record.case_id for record in first] == [record.case_id for record in second]
    assert first[0].case_id == "output_model#00000"


def test_span_truncated_default() -> None:
    """Thiếu key 'truncated' -> dùng mặc định False, không ném KeyError."""
    span = Span.from_dict({"level": "L5", "start": 0, "end": 3, "text": "abc"})
    assert span.truncated is False
