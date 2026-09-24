"""Test cho core.bio: transition IOB2, định vị token, chuyển đổi span ↔ BIO."""

from feedback.core.bio import (invalid_tags, locate_tokens, spans_from_bio, tokens_and_bio,
                               transition_errors)
from feedback.core.schemas import Span


def test_transition_rules() -> None:
    assert transition_errors(["B-L5", "I-L5", "O", "B-L3", "I-L3"]) == []
    assert transition_errors(["O", "I-L5"]) == [1]
    assert transition_errors(["B-L5", "I-L3"]) == [1]
    assert transition_errors(["I-L4"]) == [0]


def test_only_l_tags_are_valid() -> None:
    assert invalid_tags(["B-STREET", "O", "I-L8", "B-L7"]) == [0, 2]


def test_locate_tokens_rejects_dropped_text() -> None:
    assert locate_tokens("62 - 64 Hai", ["62", "-", "64", "Hai"]) == [(0, 2), (3, 4), (5, 7), (8, 11)]
    assert locate_tokens("62 - 64 Hai", ["62", "64", "Hai"]) is None      # bỏ mất "-"
    assert locate_tokens("abc", ["abc", "d"]) is None


def test_spans_from_bio_keeps_inner_whitespace() -> None:
    text = "125/ 84 Lê Lợi"
    spans = spans_from_bio(text, ["125/", "84", "Lê", "Lợi"], ["B-L6", "I-L6", "B-L5", "I-L5"])
    assert [(s.text, s.level) for s in spans] == [("125/ 84", "L6"), ("Lê Lợi", "L5")]


def test_abbrev_dot_is_separate_token_inside_span() -> None:
    text = "q. 1"
    spans = [Span(0, 4, "L3", "q. 1")]
    tokens, bio = tokens_and_bio(text, spans)
    assert tokens == ["q", ".", "1"]
    assert bio == ["B-L3", "I-L3", "I-L3"]
    assert spans_from_bio(text, tokens, bio) == spans


def test_round_trip_range_house_number() -> None:
    text = "62 - 64 Hai Bà Trưng"
    spans = [Span(0, 7, "L6", "62 - 64"), Span(8, 20, "L5", "Hai Bà Trưng")]
    tokens, bio = tokens_and_bio(text, spans)
    assert spans_from_bio(text, tokens, bio) == spans


def test_empty_spans_all_o() -> None:
    tokens, bio = tokens_and_bio("ABC XYZ", [])
    assert bio == ["O", "O"] and spans_from_bio("ABC XYZ", tokens, bio) == []
