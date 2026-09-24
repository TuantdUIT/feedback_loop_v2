"""Trạng thái của pipeline trên cùng file SQLite với broker: run, case, sổ ngân sách, hiệu chuẩn.

``cases.record`` là JSON của ``CaseRecord`` — nguồn sự thật duy nhất của một case. Message của broker
chỉ mang ``run_id`` + ``case_id``. Mọi thay đổi case đi cùng ``Broker.ack`` qua ``case_statements``
để việc ghi kết quả và việc chuyển sang giai đoạn sau là một transaction.

Chi phí của run được cộng ngay khi một request API trả về (``RunBudget.settle``), không đợi case xong,
nên tiền đã tiêu không bao giờ bị mất khỏi sổ kể cả khi message sau đó lỗi.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from typing import Any

from feedback.core.schemas import CaseRecord
from kafka_simulation.broker import Broker, Statement


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    n_cases     INTEGER NOT NULL,
    budget_usd  REAL NOT NULL,
    cost_usd    REAL NOT NULL DEFAULT 0,
    meta        TEXT NOT NULL,
    report      TEXT,
    created_at  REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS cases (
    run_id     TEXT NOT NULL,
    case_id    TEXT NOT NULL,
    position   INTEGER NOT NULL,
    stage      TEXT NOT NULL,
    decision   TEXT,
    record     TEXT NOT NULL,
    cost_usd   REAL NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    PRIMARY KEY (run_id, case_id)
);
CREATE TABLE IF NOT EXISTS calibrations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    metrics     TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""

TERMINAL_STAGE = "decided"


class Store:
    def __init__(self, broker: Broker) -> None:
        self.broker = broker
        broker.conn.executescript(SCHEMA)
        self._reserved: dict[str, float] = defaultdict(float)
        self._budget_lock = threading.Lock()

    # ----------------------------------------------------------------- runs
    def create_run(self, run_id: str, filename: str, kind: str, records: list[CaseRecord],
                   budget_usd: float, meta: dict[str, Any], first_topic: str) -> None:
        now = time.time()
        statements: list[Statement] = [(
            "INSERT INTO runs (run_id, filename, kind, status, n_cases, budget_usd, meta, created_at)"
            " VALUES (?, ?, ?, 'running', ?, ?, ?, ?)",
            (run_id, filename, kind, len(records), budget_usd, json.dumps(meta, ensure_ascii=False), now))]
        for position, record in enumerate(records):
            statements.append((
                "INSERT INTO cases (run_id, case_id, position, stage, record, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, record.case_id, position, first_topic,
                 json.dumps(record.to_dict(), ensure_ascii=False), now)))
        self.broker.publish_many(first_topic, [(run_id, r.case_id) for r in records], statements)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self.broker.query("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        return self._run_row(rows[0]) if rows else None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.broker.query("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._run_row(row) for row in rows]

    @staticmethod
    def _run_row(row: Any) -> dict[str, Any]:
        run = dict(row)
        run["meta"] = json.loads(run["meta"])
        run["report"] = json.loads(run["report"]) if run["report"] else None
        return run

    def finish_run(self, run_id: str, report: dict[str, Any]) -> None:
        with self.broker.transaction() as conn:
            conn.execute("UPDATE runs SET status = 'done', report = ?, finished_at = ? WHERE run_id = ?",
                         (json.dumps(report, ensure_ascii=False), time.time(), run_id))

    def open_cases(self, run_id: str) -> int:
        rows = self.broker.query("SELECT COUNT(*) AS n FROM cases WHERE run_id = ? AND stage != ?",
                                 (run_id, TERMINAL_STAGE))
        return rows[0]["n"]

    # ---------------------------------------------------------------- cases
    def get_record(self, run_id: str, case_id: str) -> CaseRecord:
        rows = self.broker.query("SELECT record FROM cases WHERE run_id = ? AND case_id = ?", (run_id, case_id))
        return CaseRecord.from_dict(json.loads(rows[0]["record"]))

    def records(self, run_id: str) -> list[CaseRecord]:
        rows = self.broker.query("SELECT record FROM cases WHERE run_id = ? ORDER BY position", (run_id,))
        return [CaseRecord.from_dict(json.loads(row["record"])) for row in rows]

    def case_rows(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.broker.query(
            "SELECT case_id, position, stage, decision, cost_usd, record FROM cases WHERE run_id = ? ORDER BY position",
            (run_id,))
        return [dict(row) for row in rows]

    @staticmethod
    def case_statements(run_id: str, record: CaseRecord, stage: str) -> list[Statement]:
        decision = record.decision.value if record.decision is not None else None
        return [("UPDATE cases SET stage = ?, decision = ?, record = ?, cost_usd = ?, updated_at = ?"
                 " WHERE run_id = ? AND case_id = ?",
                 (stage, decision, json.dumps(record.to_dict(), ensure_ascii=False), record.cost_usd,
                  time.time(), run_id, record.case_id))]

    def stage_counts(self, run_id: str) -> dict[str, int]:
        rows = self.broker.query("SELECT stage, COUNT(*) AS n FROM cases WHERE run_id = ? GROUP BY stage", (run_id,))
        return {row["stage"]: row["n"] for row in rows}

    # --------------------------------------------------------------- budget
    def reserve(self, run_id: str, estimate: float) -> bool:
        with self._budget_lock:
            run = self.broker.query("SELECT cost_usd, budget_usd FROM runs WHERE run_id = ?", (run_id,))[0]
            if run["cost_usd"] + self._reserved[run_id] + estimate > run["budget_usd"]:
                return False
            self._reserved[run_id] += estimate
            return True

    def settle(self, run_id: str, estimate: float, actual: float) -> None:
        with self._budget_lock:
            self._reserved[run_id] = max(0.0, self._reserved[run_id] - estimate)
            if actual:
                with self.broker.transaction() as conn:
                    conn.execute("UPDATE runs SET cost_usd = cost_usd + ? WHERE run_id = ?", (actual, run_id))

    # ----------------------------------------------------------- calibration
    def save_calibration(self, run_id: str, fingerprint: str, metrics: dict[str, Any]) -> None:
        with self.broker.transaction() as conn:
            conn.execute("INSERT INTO calibrations (run_id, fingerprint, metrics, created_at) VALUES (?, ?, ?, ?)",
                         (run_id, fingerprint, json.dumps(metrics, ensure_ascii=False), time.time()))

    def latest_calibration(self, fingerprint: str) -> dict[str, Any] | None:
        rows = self.broker.query("SELECT * FROM calibrations WHERE fingerprint = ? ORDER BY id DESC LIMIT 1",
                                 (fingerprint,))
        if not rows:
            return None
        row = dict(rows[0])
        row["metrics"] = json.loads(row["metrics"])
        return row


class RunBudget:
    """Sổ ngân sách của một run, dùng làm ``budget`` cho ``llm_client.chat``."""

    def __init__(self, store: Store, run_id: str) -> None:
        self.store, self.run_id = store, run_id

    def reserve(self, estimate_usd: float) -> bool:
        return self.store.reserve(self.run_id, estimate_usd)

    def settle(self, estimate_usd: float, actual_usd: float) -> None:
        self.store.settle(self.run_id, estimate_usd, actual_usd)
