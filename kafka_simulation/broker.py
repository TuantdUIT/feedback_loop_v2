"""Broker lưu bền trên SQLite: topic theo giai đoạn, claim nguyên tử, ack/retry và dead-letter.

Message chỉ mang ``run_id`` và ``case_id``; dữ liệu của case nằm ở bảng khác do nơi gọi quản lý.
``ack`` nhận thêm các câu SQL để ghi dữ liệu đó trong CÙNG transaction với việc đánh dấu xong và
publish sang topic kế tiếp, nên không bao giờ có trạng thái "đã ghi kết quả nhưng chưa chuyển tiếp".

Giao hàng là at-least-once: message đang ``processing`` khi tiến trình chết sẽ được ``recover()``
trả về ``pending`` ở lần khởi động sau.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    topic        TEXT    NOT NULL,
    run_id       TEXT    NOT NULL,
    case_id      TEXT    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'pending',
    attempts     INTEGER NOT NULL DEFAULT 0,
    available_at REAL    NOT NULL,
    last_error   TEXT,
    created_at   REAL    NOT NULL,
    updated_at   REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_claim ON messages (topic, status, available_at, id);
CREATE INDEX IF NOT EXISTS ix_messages_run ON messages (run_id);
"""

Statement = tuple[str, tuple[Any, ...]]


@dataclass(frozen=True)
class Message:
    """Một message đã được claim; ``attempts`` đã tính cả lần claim này."""

    id: int
    topic: str
    run_id: str
    case_id: str
    attempts: int


class Broker:
    """Hàng đợi nhiều topic trên một file SQLite, an toàn khi nhiều luồng cùng dùng."""

    def __init__(self, path: str | Path, max_attempts: int = 3,
                 backoff_sec: Iterable[float] = (5, 20, 60)) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self.backoff_sec = tuple(backoff_sec)
        self._lock = threading.RLock()
        self._events: dict[str, threading.Event] = defaultdict(threading.Event)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.executescript(SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Một transaction ghi; rollback toàn bộ nếu có lỗi."""
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Đọc không cần transaction ghi."""
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    def _insert(self, conn: sqlite3.Connection, topic: str, run_id: str, case_id: str) -> int:
        now = time.time()
        cursor = conn.execute(
            "INSERT INTO messages (topic, run_id, case_id, available_at, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)", (topic, run_id, case_id, now, now, now))
        return int(cursor.lastrowid)

    def publish(self, topic: str, run_id: str, case_id: str,
                statements: Iterable[Statement] = ()) -> int:
        """Đưa một message vào cuối topic, kèm các câu SQL chạy cùng transaction."""
        with self.transaction() as conn:
            for sql, params in statements:
                conn.execute(sql, params)
            message_id = self._insert(conn, topic, run_id, case_id)
        self._events[topic].set()
        return message_id

    def publish_many(self, topic: str, items: Iterable[tuple[str, str]],
                     statements: Iterable[Statement] = ()) -> int:
        """Publish nhiều message trong một transaction; trả số message đã thêm."""
        with self.transaction() as conn:
            for sql, params in statements:
                conn.execute(sql, params)
            count = 0
            for run_id, case_id in items:
                self._insert(conn, topic, run_id, case_id)
                count += 1
        self._events[topic].set()
        return count

    def claim(self, topic: str) -> Message | None:
        """Lấy message pending cũ nhất đã tới hạn của topic, chuyển sang processing."""
        now = time.time()
        with self.transaction() as conn:
            row = conn.execute(
                "UPDATE messages SET status = 'processing', attempts = attempts + 1, updated_at = ?"
                " WHERE id = (SELECT id FROM messages WHERE topic = ? AND status = 'pending'"
                "             AND available_at <= ? ORDER BY id LIMIT 1)"
                " RETURNING id, topic, run_id, case_id, attempts", (now, topic, now)).fetchone()
        if row is None:
            return None
        return Message(id=row["id"], topic=row["topic"], run_id=row["run_id"],
                       case_id=row["case_id"], attempts=row["attempts"])

    def ack(self, message: Message, next_topic: str | None = None,
            statements: Iterable[Statement] = ()) -> None:
        """Đánh dấu xong, chạy các câu SQL đi kèm và publish sang topic kế tiếp — nguyên tử."""
        with self.transaction() as conn:
            for sql, params in statements:
                conn.execute(sql, params)
            conn.execute("UPDATE messages SET status = 'done', updated_at = ? WHERE id = ?",
                         (time.time(), message.id))
            if next_topic is not None:
                self._insert(conn, next_topic, message.run_id, message.case_id)
        if next_topic is not None:
            self._events[next_topic].set()

    def nack(self, message: Message, error: str, retry: bool = True) -> bool:
        """Trả message về hàng với backoff; hết lượt thử thì chuyển failed. True nếu còn được thử lại."""
        requeue = retry and message.attempts < self.max_attempts
        now = time.time()
        with self.transaction() as conn:
            if requeue:
                delay = self.backoff_sec[min(message.attempts - 1, len(self.backoff_sec) - 1)]
                conn.execute("UPDATE messages SET status = 'pending', available_at = ?, last_error = ?,"
                             " updated_at = ? WHERE id = ?", (now + delay, error[:500], now, message.id))
            else:
                conn.execute("UPDATE messages SET status = 'failed', last_error = ?, updated_at = ?"
                             " WHERE id = ?", (error[:500], now, message.id))
        return requeue

    def recover(self) -> int:
        """Trả mọi message processing (tiến trình trước chết giữa chừng) về pending."""
        with self.transaction() as conn:
            cursor = conn.execute("UPDATE messages SET status = 'pending', available_at = ?, updated_at = ?"
                                  " WHERE status = 'processing'", (time.time(), time.time()))
        return cursor.rowcount

    def wait(self, topic: str, timeout: float) -> None:
        """Ngủ tới khi topic có message mới hoặc hết timeout."""
        event = self._events[topic]
        event.wait(timeout)
        event.clear()

    def notify(self, topic: str) -> None:
        """Đánh thức worker của topic, ví dụ khi dừng pipeline."""
        self._events[topic].set()

    def counts(self, run_id: str | None = None) -> dict[str, dict[str, int]]:
        """Số message theo topic và trạng thái, lọc theo run nếu có."""
        sql = "SELECT topic, status, COUNT(*) AS n FROM messages"
        params: tuple[Any, ...] = ()
        if run_id is not None:
            sql += " WHERE run_id = ?"
            params = (run_id,)
        result: dict[str, dict[str, int]] = defaultdict(dict)
        for row in self.query(sql + " GROUP BY topic, status", params):
            result[row["topic"]][row["status"]] = row["n"]
        return dict(result)

    def close(self) -> None:
        with self._lock:
            self.conn.close()
