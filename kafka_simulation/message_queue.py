"""Hàng đợi FIFO trong bộ nhớ, mô phỏng message queue giữa producer và consumer."""

from __future__ import annotations

from collections import deque
from typing import Any


class MessageQueue:
    """Producer `publish()` vào cuối hàng, consumer `consume()` lấy ra từ đầu hàng."""

    def __init__(self, name: str = "ner.raw") -> None:
        self.name = name
        self._items: deque[dict[str, Any]] = deque()
        self.published = 0
        self.consumed = 0

    def publish(self, message: dict[str, Any]) -> None:
        """Đưa một message vào cuối hàng đợi."""
        self._items.append(message)
        self.published += 1

    def consume(self) -> dict[str, Any] | None:
        """Lấy message đầu hàng; trả None khi hàng đợi rỗng."""
        if not self._items:
            return None
        self.consumed += 1
        return self._items.popleft()

    def __len__(self) -> int:
        return len(self._items)
