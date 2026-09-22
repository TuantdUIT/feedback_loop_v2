"""Định nghĩa các kiểu dữ liệu thuần dùng chung cho bốn lớp."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Self


@dataclass(frozen=True, order=True)
class Span:
    """Một thực thể đã gộp, định vị trên chuỗi văn bản NFC."""

    start: int
    end: int
    level: str
    text: str
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Chuyển span về đúng thứ tự key của JSON pipeline."""
        return {
            "level": self.level,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Tạo span từ dictionary của dữ liệu pipeline."""
        return cls(
            start=data["start"],
            end=data["end"],
            level=data["level"],
            text=data["text"],
            truncated=data.get("truncated", False),
        )


@dataclass
class ParseResult:
    """Kết quả parse tại một bước của feedback loop."""

    tokens: list[str]
    bio: list[str]
    spans: list[Span]
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Chuyển kết quả parse thành dictionary tuần tự hoá được."""
        return {
            "tokens": self.tokens,
            "bio": self.bio,
            "spans": [span.to_dict() for span in self.spans],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Tạo kết quả parse từ dictionary."""
        return cls(
            tokens=list(data["tokens"]),
            bio=list(data["bio"]),
            spans=[Span.from_dict(span) for span in data["spans"]],
            source=data["source"],
        )


@dataclass(frozen=True)
class RunMeta:
    """Thông tin truy vết phiên chạy và phiên bản prompt."""

    prompt_sha: str
    prompt_variant: str
    git_commit: str
    run_at: str

    def to_dict(self) -> dict[str, str]:
        """Chuyển metadata phiên chạy thành dictionary."""
        return {
            "prompt_sha": self.prompt_sha,
            "prompt_variant": self.prompt_variant,
            "git_commit": self.git_commit,
            "run_at": self.run_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Tạo metadata phiên chạy từ dictionary."""
        return cls(
            prompt_sha=data["prompt_sha"],
            prompt_variant=data["prompt_variant"],
            git_commit=data["git_commit"],
            run_at=data["run_at"],
        )


class Decision(str, Enum):
    """Các quyết định cuối cùng mà cascade có thể đưa ra."""

    KEEP_OLD = "keep_old"
    ACCEPT_NEW = "accept_new"
    REWRITE = "rewrite"
    ESCALATE_HUMAN = "escalate_human"

    def to_dict(self) -> str:
        """Trả giá trị chuỗi dùng trực tiếp trong JSON."""
        return self.value

    @classmethod
    def from_dict(cls, data: str) -> Self:
        """Tạo quyết định từ giá trị chuỗi trong JSON."""
        return cls(data)


@dataclass
class CaseRecord:
    """Một case cùng toàn bộ trace được bổ sung tuần tự qua các lớp."""

    case_id: str
    raw_text: str
    old: ParseResult
    meta: dict[str, Any]
    new: ParseResult | None = None
    layers: dict[str, dict[str, Any]] = field(default_factory=dict)
    diff: dict[str, Any] = field(default_factory=dict)
    decision: Decision | None = None
    cost_usd: float = 0.0

    def attach(self, layer: str, payload: dict[str, Any]) -> None:
        """Gắn kết quả của một lớp; từ chối ghi đè trace đã tồn tại."""
        if layer in self.layers:
            raise ValueError(f"Lớp {layer!r} đã có dữ liệu")
        self.layers[layer] = payload

    def to_dict(self) -> dict[str, Any]:
        """Chuyển case cùng toàn bộ trace thành dictionary."""
        return {
            "case_id": self.case_id,
            "raw_text": self.raw_text,
            "old": self.old.to_dict(),
            "meta": self.meta,
            "new": self.new.to_dict() if self.new is not None else None,
            "layers": self.layers,
            "diff": self.diff,
            "decision": self.decision.to_dict() if self.decision is not None else None,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Khôi phục case cùng toàn bộ trace từ dictionary."""
        new_data = data.get("new")
        decision_data = data.get("decision")
        return cls(
            case_id=data["case_id"],
            raw_text=data["raw_text"],
            old=ParseResult.from_dict(data["old"]),
            meta=dict(data["meta"]),
            new=ParseResult.from_dict(new_data) if new_data is not None else None,
            layers=dict(data.get("layers", {})),
            diff=dict(data.get("diff", {})),
            decision=Decision.from_dict(decision_data) if decision_data is not None else None,
            cost_usd=data.get("cost_usd", 0.0),
        )
