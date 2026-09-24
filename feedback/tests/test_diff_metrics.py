"""Test cho core.diff_metrics: COR/INC/PAR/MIS/SPU, truncated, abstain, severity."""

from feedback.core.diff_metrics import compare
from feedback.core.schemas import Span

W = {"L1": 0.3, "L2": 1.0, "L3": 1.0, "L4": 0.7, "L5": 0.8, "L6": 0.9, "L7": 0.5}


def test_identical() -> None:
    spans = [Span(0, 2, "L6", "74"), Span(3, 16, "L5", "Nơ Trang Long")]
    result = compare(spans, list(spans), W)
    assert result["identical"] and result["counts"]["COR"] == 2 and result["severity_total"] == 0


def test_label_error_is_inc_with_max_weight() -> None:
    result = compare([Span(0, 7, "L4", "Tân Phú")], [Span(0, 7, "L3", "Tân Phú")], W)
    assert result["counts"]["INC"] == 1
    assert result["severity_total"] == 1.0


def test_boundary_error_is_par() -> None:
    old = [Span(0, 24, "L4", "Phường Tân Thạnh Tân Phú")]
    new = [Span(0, 16, "L4", "Phường Tân Thạnh"), Span(17, 24, "L3", "Tân Phú")]
    result = compare(old, new, W)
    assert result["counts"] == {"COR": 0, "INC": 0, "PAR": 1, "MIS": 1, "SPU": 0, "TRUNC": 0}
    par = next(i for i in result["items"] if i["kind"] == "PAR")
    assert par["label_match"] is True and par["new"]["text"] == "Phường Tân Thạnh"
    assert result["levels_involved"] == ["L3", "L4"]


def test_missing_and_spurious() -> None:
    result = compare([Span(0, 3, "L6", "123")], [Span(4, 10, "L5", "Lê Lợi")], W)
    assert result["counts"]["SPU"] == 1 and result["counts"]["MIS"] == 1


def test_truncated_ignored_by_default() -> None:
    old = [Span(0, 5, "L5", "Đường", truncated=False)]
    new = [Span(0, 5, "L5", "Đường", truncated=True)]
    assert compare(old, new, W)["identical"] is True
    strict = compare(old, new, W, ignore_truncated=False)
    assert strict["identical"] is False and strict["counts"]["TRUNC"] == 1


def test_abstain_flags() -> None:
    assert compare([], [Span(0, 3, "L6", "123")], W)["abstain"] == "old_empty"
    assert compare([Span(0, 3, "L6", "123")], [], W)["abstain"] == "new_empty"
    both_empty = compare([], [], W)
    assert both_empty["identical"] and both_empty["abstain"] is None
