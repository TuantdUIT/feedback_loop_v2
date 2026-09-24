"""Pipeline theo topic: upload → deepseek_parse → compare → qwen_judge → gate → finalize.

Không có orchestrator gọi tuần tự: mỗi giai đoạn là một nhóm worker chỉ đọc topic của mình và
publish sang topic kế tiếp khi xong (trigger dây chuyền). ``Broker.ack`` ghi kết quả case và chuyển
tiếp trong cùng một transaction.

Xử lý lỗi của một message:
  BudgetExceeded     case kết thúc ``budget_exceeded``, không gọi API nữa.
  FatalAPIError      case kết thúc ``failed`` (sai key, hết tiền, sai tham số).
  lỗi khác           broker retry với backoff; hết lượt thử → case kết thúc ``failed``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from feedback.core import llm_client
from feedback.core.config import PROJECT_ROOT, load_yaml
from feedback.core.normalize import make_case_id
from feedback.core.prompt_loader import build_run_meta
from feedback.core.schemas import CaseRecord, Decision
from feedback.eval import report
from feedback.pipeline.store import RunBudget, Store
from feedback.stages import compare, deepseek_parser, gate, qwen_judge
from feedback.stages.validator import validate_parse
from kafka_simulation.bridge import to_case_record
from kafka_simulation.broker import Broker, Message
from kafka_simulation.capture_parser import parse_capture
from kafka_simulation.message import validate

DEFAULT_DB = PROJECT_ROOT / "feedback" / "store" / "pipeline.db"
UPLOAD_DIR = PROJECT_ROOT / "feedback" / "store" / "uploads"
TOPICS = ("ingested", "parsed", "to_judge", "agreed", "judged", "decided")


@dataclass
class Stage:
    name: str
    topics: tuple[str, ...]
    workers: int
    fn: Callable[[CaseRecord, RunBudget], tuple[CaseRecord, str]]


class Pipeline:
    def __init__(self, db_path: str | Path = DEFAULT_DB, upload_dir: str | Path = UPLOAD_DIR) -> None:
        thresholds = load_yaml("thresholds")
        queue = thresholds.get("queue") or {}
        self.broker = Broker(db_path, max_attempts=queue.get("max_attempts", 3),
                             backoff_sec=queue.get("backoff_sec", (5, 20, 60)))
        self.store = Store(self.broker)
        self.upload_dir = Path(upload_dir)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._finalize_lock = threading.Lock()
        models = load_yaml("models")
        self.stages = [
            Stage("deepseek_parse", ("ingested",), models["deepseek"].get("workers", 4), self._deepseek),
            Stage("compare", ("parsed",), 1, lambda record, budget: compare.run(record)),
            Stage("qwen_judge", ("to_judge",), models["qwen"].get("workers", 4), qwen_judge.run),
            Stage("gate", ("agreed", "judged"), 1,
                  lambda record, budget: gate.run(record, load_yaml("thresholds"))),
        ]

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self.broker.recover()
        self._stop.clear()
        for stage in self.stages:
            for index in range(stage.workers):
                self._spawn(f"{stage.name}-{index}", lambda s=stage: self._loop(s.topics, lambda m, s=s: self._handle(s, m)))
        self._spawn("finalize", lambda: self._loop(("decided",), self._finalize))

    def _spawn(self, name: str, target: Callable[[], None]) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for topic in TOPICS:
            self.broker.notify(topic)
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads.clear()

    def _loop(self, topics: tuple[str, ...], handle: Callable[[Message], None]) -> None:
        while not self._stop.is_set():
            message = next((m for m in (self.broker.claim(t) for t in topics) if m is not None), None)
            if message is None:
                self.broker.wait(topics[0], 0.5)
                continue
            try:
                handle(message)
            except Exception:                              # lỗi của chính pipeline, không để worker chết
                self.broker.nack(message, traceback.format_exc()[-500:])

    # ---------------------------------------------------------------- stages
    def _deepseek(self, record: CaseRecord, budget: RunBudget) -> tuple[CaseRecord, str]:
        retry = (load_yaml("thresholds").get("deepseek") or {}).get("retry_on_hard_violation", 1)
        return deepseek_parser.parse(record, budget, retry_on_hard=retry)

    def _handle(self, stage: Stage, message: Message) -> None:
        record = self.store.get_record(message.run_id, message.case_id)
        budget = RunBudget(self.store, message.run_id)
        try:
            record, next_topic = stage.fn(record, budget)
        except llm_client.BudgetExceeded as exc:
            record.cost_usd += exc.cost_usd
            record.decision = Decision.BUDGET_EXCEEDED
            record.layers[f"{stage.name}_error"] = {"type": "budget", "message": str(exc)}
            next_topic = "decided"
        except llm_client.FatalAPIError as exc:
            record.cost_usd += exc.cost_usd
            record.decision = Decision.FAILED
            record.layers[f"{stage.name}_error"] = {"type": "fatal", "message": str(exc), "attempts": exc.attempts}
            next_topic = "decided"
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if message.attempts < self.broker.max_attempts:
                self.broker.nack(message, detail)
                return
            record.decision = Decision.FAILED
            record.layers[f"{stage.name}_error"] = {"type": "exhausted", "message": detail[:500],
                                                    "attempts": getattr(exc, "attempts", [])}
            next_topic = "decided"
        self.broker.ack(message, next_topic, self.store.case_statements(message.run_id, record, next_topic))

    def _finalize(self, message: Message) -> None:
        self.broker.ack(message)
        with self._finalize_lock:
            run = self.store.get_run(message.run_id)
            if run is None or run["status"] != "running" or self.store.open_cases(message.run_id):
                return
            self.store.finish_run(message.run_id, self.build_report(message.run_id))

    def build_report(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        records = self.store.records(run_id)
        fingerprint = run["meta"].get("judge_fingerprint")
        if run["kind"] == "calibration":
            metrics = report.calibration_metrics(records)
            if metrics is not None and fingerprint:
                self.store.save_calibration(run_id, fingerprint, metrics)
        calibration = self.store.latest_calibration(fingerprint) if fingerprint else None
        return report.build_report(records, load_yaml("thresholds"), calibration)

    # ---------------------------------------------------------------- ingest
    def _load(self, filename: str, content: str, gold: dict[str, Any] | None,
              run_id: str) -> tuple[list[CaseRecord], list[dict[str, Any]]]:
        """Đọc capture, validate message, dựng CaseRecord + Lớp 0 cho bản parse của model."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        path = self.upload_dir / f"{run_id}__{Path(filename).name}"
        path.write_text(content, encoding="utf-8")
        try:
            payloads = parse_capture(path)
        except ValueError as exc:                          # bỏ đường dẫn trên máy khỏi thông báo
            raise ValueError(str(exc).replace(str(path), Path(filename).name)) from exc
        gold_by_text = {}
        if gold is not None:
            gold_by_text = {t: r["result"] for t, r in zip(gold["texts"], gold["results"], strict=True)}
        audit_rate = (load_yaml("thresholds").get("audit") or {}).get("rate", 0.0)
        now = datetime.now().astimezone().isoformat(timespec="seconds")

        records, skipped = [], []
        for payload in payloads:
            index = payload["source"]["batch_index"]
            msg = {"msg_id": make_case_id(filename, index), "text": payload["text"], "result": payload["result"],
                   "source": payload["source"], "produced_at": now}
            errors = validate(msg)
            if errors:
                skipped.append({"batch_index": index, "text": payload["text"], "errors": errors})
                continue
            record = to_case_record(msg)
            record.meta["batch_index"] = index
            digest = hashlib.sha256(f"{run_id}|{record.case_id}".encode()).digest()
            record.meta["audit"] = int.from_bytes(digest[:4], "big") / 2**32 < audit_rate
            if payload["text"] in gold_by_text:
                record.meta["gold"] = gold_by_text[payload["text"]]
            layer0 = validate_parse(record.raw_text, record.old.spans, record.old.tokens, record.old.bio)
            unlocated = [{"rule": "text_not_in_input", **item} for item in record.meta.get("unlocated", [])]
            record.layers["layer0_model"] = {"ok": layer0["ok"] and not unlocated,
                                             "hard": unlocated + layer0["hard"], "soft": layer0["soft"]}
            records.append(record)
        return records, skipped

    def estimate(self, n_cases: int) -> dict[str, float]:
        """Trần chi phí ước tính: giả định mọi case đều cần judge (thực tế thấp hơn nhờ nhánh agreed)."""
        models = load_yaml("models")
        ds = n_cases * models["deepseek"]["estimate_usd_per_call"]
        judge = n_cases * 2 * models["qwen"]["estimate_usd_per_call"]
        return {"deepseek_usd": round(ds, 4), "judge_max_usd": round(judge, 4), "total_max_usd": round(ds + judge, 4)}

    def preview(self, filename: str, content: str) -> dict[str, Any]:
        records, skipped = self._load(filename, content, None, "preview")
        budget = (load_yaml("thresholds").get("budget") or {}).get("per_run_usd", 2.0)
        return {"filename": filename, "n_cases": len(records), "skipped": skipped,
                "estimate": self.estimate(len(records)), "budget_usd": budget,
                "model_layer0_hard": sum(1 for r in records if not r.layers["layer0_model"]["ok"]),
                "sample": [{"case_id": r.case_id, "text": r.raw_text,
                            "spans": [s.to_dict() for s in r.old.spans]} for r in records[:5]]}

    def submit(self, filename: str, content: str, kind: str = "normal",
               gold: dict[str, Any] | None = None) -> str:
        if kind not in ("normal", "calibration"):
            raise ValueError("kind phải là normal hoặc calibration")
        if kind == "calibration" and gold is None:
            raise ValueError("Lượt hiệu chuẩn cần file gold")
        run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        records, skipped = self._load(filename, content, gold, run_id)
        if not records:
            raise ValueError("File không có case hợp lệ nào")
        fingerprint, parts = qwen_judge.fingerprint()
        thresholds = load_yaml("thresholds")
        meta = {**build_run_meta().to_dict(), "judge_fingerprint": fingerprint, "judge_config": parts,
                "thresholds_version": thresholds.get("version"), "skipped": skipped,
                "estimate": self.estimate(len(records)),
                "n_gold": sum(1 for r in records if "gold" in r.meta)}
        budget = (thresholds.get("budget") or {}).get("per_run_usd", 2.0)
        self.store.create_run(run_id, filename, kind, records, budget, meta, first_topic="ingested")
        return run_id

    # ---------------------------------------------------------------- queries
    def progress(self, run_id: str) -> dict[str, Any]:
        messages = self.broker.counts(run_id)
        processing = {topic: counts.get("processing", 0) for topic, counts in messages.items()}
        failed = {topic: counts.get("failed", 0) for topic, counts in messages.items() if counts.get("failed")}
        return {"stages": self.store.stage_counts(run_id), "processing": processing, "failed_messages": failed}

    def wait(self, run_id: str, timeout: float = 3600, poll: float = 1.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = self.store.get_run(run_id)
            if run["status"] == "done":
                return run
            time.sleep(poll)
        raise TimeoutError(f"{run_id} chưa xong sau {timeout}s")
