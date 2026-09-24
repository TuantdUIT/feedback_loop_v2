"""So sánh hai bản parse và phân loại từng điểm khác biệt.

Phân loại theo từng cặp span (``old`` = model đang triển khai, ``new`` = DeepSeek):

  COR  cùng biên, cùng level
  INC  cùng biên, khác level                     (lỗi nhãn)
  PAR  biên chồng lấn nhưng không trùng          (lỗi biên; ``label_match`` cho biết level có khớp)
  MIS  chỉ có ở ``new``                          (model thiếu so với DeepSeek)
  SPU  chỉ có ở ``old``                          (model thừa so với DeepSeek)

``new`` chỉ là mốc để gọi tên, KHÔNG phải đáp án đúng — bên nào đúng là việc của judge.
``truncated`` mặc định bị bỏ qua vì model đang triển khai không có tín hiệu này
(``meta.truncated_unknown``). ``spans`` rỗng có thể là đáp án đúng, nên trường hợp một bên rỗng được
đánh dấu ``abstain`` riêng. Module không tự quyết gate; severity chỉ là số đo.
"""

from __future__ import annotations

from typing import Any

from feedback.core.schemas import Span


DEFAULT_WEIGHTS = {"L1": 0.3, "L2": 1.0, "L3": 1.0, "L4": 0.7, "L5": 0.8, "L6": 0.9, "L7": 0.5}


def _overlap(a: Span, b: Span) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def _brief(span: Span | None) -> dict[str, Any] | None:
    if span is None:
        return None
    return {"level": span.level, "text": span.text, "start": span.start, "end": span.end}


def compare(old: list[Span], new: list[Span], weights: dict[str, float] | None = None,
            ignore_truncated: bool = True) -> dict[str, Any]:
    """Ghép cặp span hai bên và trả danh sách khác biệt, số đếm và severity."""
    w = weights or DEFAULT_WEIGHTS
    items: list[dict[str, Any]] = []
    old_left = sorted(old)
    new_left = sorted(new)

    for n in list(new_left):
        same = next((o for o in old_left if (o.start, o.end) == (n.start, n.end)), None)
        if same is None:
            continue
        old_left.remove(same)
        new_left.remove(n)
        if same.level == n.level:
            kind = "COR"
            severity = 0.0
            if not ignore_truncated and same.truncated != n.truncated:
                kind, severity = "TRUNC", 0.1
        else:
            kind, severity = "INC", max(w.get(same.level, 1.0), w.get(n.level, 1.0))
        items.append({"kind": kind, "old": _brief(same), "new": _brief(n),
                      "label_match": same.level == n.level, "severity": severity})

    pairs = sorted(((_overlap(o, n), o, n) for o in old_left for n in new_left if _overlap(o, n)),
                   key=lambda item: -item[0])
    for _, o, n in pairs:
        if o not in old_left or n not in new_left:
            continue
        old_left.remove(o)
        new_left.remove(n)
        match = o.level == n.level
        severity = w.get(n.level, 1.0) if match else max(w.get(o.level, 1.0), w.get(n.level, 1.0))
        items.append({"kind": "PAR", "old": _brief(o), "new": _brief(n), "label_match": match,
                      "severity": severity})

    items += [{"kind": "MIS", "old": None, "new": _brief(n), "label_match": False,
               "severity": w.get(n.level, 1.0)} for n in new_left]
    items += [{"kind": "SPU", "old": _brief(o), "new": None, "label_match": False,
               "severity": w.get(o.level, 1.0)} for o in old_left]
    items.sort(key=lambda item: (item["old"] or item["new"])["start"])

    counts = {kind: sum(item["kind"] == kind for item in items)
              for kind in ("COR", "INC", "PAR", "MIS", "SPU", "TRUNC")}
    differences = [item for item in items if item["kind"] != "COR"]
    abstain = None
    if not old and new:
        abstain = "old_empty"
    elif old and not new:
        abstain = "new_empty"
    return {
        "identical": not differences,
        "items": items,
        "counts": counts,
        "n_old": len(old),
        "n_new": len(new),
        "severity_total": round(sum(item["severity"] for item in differences), 4),
        "severity_max": max((item["severity"] for item in differences), default=0.0),
        "levels_involved": sorted({span["level"] for item in differences
                                   for span in (item["old"], item["new"]) if span}),
        "abstain": abstain,
    }
