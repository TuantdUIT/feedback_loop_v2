"""Kiểm thử bridge trên đủ 52 mẫu thật, gồm NFD và địa chỉ dính."""

import json
from pathlib import Path

import pytest

from feedback.core.normalize import make_case_id
from feedback.core.schemas import CaseRecord
from kafka_simulation.bridge import to_case_record


FIXTURE = Path(__file__).parent / "fixtures/template_52.json"


@pytest.fixture(scope="module")
def records() -> list[CaseRecord]:
    """Fixture do capture_parser sinh ra; riêng marker NFD được sửa trong fixture."""
    payloads = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert len(payloads) == 52
    return [to_case_record({"msg_id": make_case_id("template.txt", index), **payload})
            for index, payload in enumerate(payloads)]


def test_all_70_spans_located(records: list[CaseRecord]) -> None:
    assert sum(len(record.old.spans) for record in records) == 70
    assert all(not record.meta.get("unlocated") for record in records)


def test_nfd_shift_is_flagged(records: list[CaseRecord]) -> None:
    assert records[22].meta["nfc_shift"] is True
    assert records[22].raw_text.startswith("Nguyễn Huệ")
    assert all(record.meta.get("nfc_shift") is None for index, record in enumerate(records) if index != 22)


def test_loose_whitespace_match_keeps_actual_slice(records: list[CaseRecord]) -> None:
    record = records[44]
    span = next(span for span in record.old.spans if span.level == "L3")
    assert (span.start, span.end, span.text) == (21, 31, "1Đức Chính")
    assert record.meta["loose_ws_match"] == [{"level": "L3", "text": "1 Đức Chính"}]


def test_tokens_never_cross_span_boundaries(records: list[CaseRecord]) -> None:
    for record in records:
        assert len(record.old.tokens) == len(record.old.bio)
        for span in record.old.spans:
            assert span.text == record.raw_text[span.start:span.end]
            assert any(label == "B-" + span.level for label in record.old.bio)
        # Mỗi token được tạo từ các điểm cắt có biên span; không thể chồng lấn một phần.
        position = 0
        for token in record.old.tokens:
            start = record.raw_text.find(token, position)
            assert start >= 0
            end = start + len(token)
            for span in record.old.spans:
                assert not (start < span.end and end > span.start) or (span.start <= start and end <= span.end)
            position = end


def test_expected_token_and_bio_examples(records: list[CaseRecord]) -> None:
    city = records[8]
    assert city.old.tokens == ["Thành", "phố", "Hồ", "Chí", "Minh", ",", "Quận", "1"]
    assert city.old.bio == ["B-L2", "I-L2", "I-L2", "I-L2", "I-L2", "O", "B-L3", "I-L3"]
    street = records[4]
    assert street.old.tokens == ["74", ",", "Phường", "12", "Nơ", "Trang", "Long"]
    assert street.old.bio == ["B-L6", "O", "B-L4", "I-L4", "O", "O", "O"]


def test_empty_spans_truncated_and_roundtrip(records: list[CaseRecord]) -> None:
    assert sum(not record.old.spans for record in records) == 7
    for record in records:
        assert record.old.source == "gsm_api"
        assert record.meta["truncated_unknown"] is True
        assert all(span.truncated is False for span in record.old.spans)
        assert CaseRecord.from_dict(record.to_dict()) == record


def test_unlocated_span_is_dropped_without_changing_other_data() -> None:
    message = {
        "msg_id": "synthetic#00000", "text": "Nam Kỳ Khởi Nghĩa",
        "confidence": 0.9,
        "result": {f"L{i}": [] for i in range(1, 8)},
    }
    message["result"]["L5"] = ["Không tồn tại", "Nam"]
    record = to_case_record(message)
    assert record.meta["unlocated"] == [{"level": "L5", "text": "Không tồn tại"}]
    assert [span.text for span in record.old.spans] == ["Nam"]
    assert record.old.bio[0] == "B-L5"
