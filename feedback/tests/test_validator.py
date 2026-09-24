"""Test cho stages.validator (Lớp 0): hard ở span, soft ở BIO, tự sửa offset khi chắc chắn."""

from feedback.core.bio import tokens_and_bio
from feedback.core.schemas import Span
from feedback.stages.validator import validate_parse

TEXT = "74 Nơ Trang Long, Phường 12"


def test_valid_parse_passes() -> None:
    spans = [Span(0, 2, "L6", "74"), Span(3, 16, "L5", "Nơ Trang Long"), Span(18, 27, "L4", "Phường 12")]
    tokens, bio = tokens_and_bio(TEXT, spans)
    result = validate_parse(TEXT, spans, tokens, bio)
    assert result["ok"] and result["hard"] == [] and result["soft"] == []


def test_offset_repaired_when_unique() -> None:
    result = validate_parse(TEXT, [Span(0, 9, "L4", "Phường 12")])
    assert result["ok"]
    assert result["spans"] == [Span(18, 27, "L4", "Phường 12")]
    assert result["soft"][0]["rule"] == "offset_repaired"


def test_offset_ambiguous_is_hard() -> None:
    result = validate_parse("Quận 1 Quận 1", [Span(3, 9, "L3", "Quận 1")])
    assert not result["ok"] and result["hard"][0]["rule"] == "offset_ambiguous"


def test_text_not_in_input_level_invalid_and_overlap_are_hard() -> None:
    result = validate_parse(TEXT, [Span(0, 2, "L6", "75"), Span(0, 2, "L9", "74")])
    assert {v["rule"] for v in result["hard"]} == {"text_not_in_input", "level_invalid"}
    overlap = validate_parse(TEXT, [Span(3, 16, "L5", "Nơ Trang Long"), Span(6, 16, "L5", "Trang Long")])
    assert [v["rule"] for v in overlap["hard"]] == ["overlap"]


def test_bio_problems_are_soft_only() -> None:
    spans = [Span(0, 2, "L6", "74")]
    tokens = ["74", "Nơ", "Trang", "Long", ",", "Phường", "12"]
    bio = ["B-L6", "I-L5", "I-L5", "I-L5", "O", "B-STREET", "O"]
    result = validate_parse(TEXT, spans, tokens, bio)
    assert result["ok"]
    assert {v["rule"] for v in result["soft"]} >= {"tag_invalid", "iob2_transition"}
