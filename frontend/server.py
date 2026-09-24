"""Local comparison UI + pipeline dashboard. Start with python -m uvicorn frontend.server:app --port 8000.

Worker của pipeline (feedback/pipeline) chạy trong cùng tiến trình, khởi động trong lifespan.
Đặt PIPELINE_DISABLED=1 để chỉ bật trang so sánh parser.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from frontend import engines


STATIC = Path(__file__).resolve().parent / "static"
pipeline = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global pipeline
    if os.environ.get("PIPELINE_DISABLED") != "1":
        from feedback.pipeline.runner import Pipeline

        pipeline = Pipeline()
        pipeline.start()
    yield
    if pipeline is not None:
        pipeline.stop()


app = FastAPI(title="So sánh parser địa chỉ", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ParseRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def _text(request: ParseRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Địa chỉ không được để trống")
    return text


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    memory = {"process_mb": None, "available_mb": None, "total_mb": None}
    try:
        import psutil

        system = psutil.virtual_memory()
        memory = {
            "process_mb": round(psutil.Process(os.getpid()).memory_info().rss / 1048576),
            "available_mb": round(system.available / 1048576),
            "total_mb": round(system.total / 1048576),
        }
    except ImportError:
        pass
    return {"loaded": engines.loaded_status(), "memory": memory,
            "openrouter_configured": engines.qwen_configured()}


@app.post("/api/parse/pytorch")
def parse_pytorch(request: ParseRequest):
    return engines.parse_pytorch(_text(request))


@app.post("/api/parse/onnx")
def parse_onnx(request: ParseRequest):
    return engines.parse_onnx(_text(request))


@app.post("/api/parse/qwen")
def parse_qwen(request: ParseRequest):
    return engines.parse_qwen(_text(request))


# ------------------------------------------------------------------ pipeline
class UploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=5_000_000)


class RunRequest(UploadRequest):
    kind: Literal["normal", "calibration"] = "normal"
    gold: dict[str, Any] | None = None


def _pipeline():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline đang tắt (PIPELINE_DISABLED=1)")
    return pipeline


def _run_or_404(run_id: str) -> dict[str, Any]:
    run = _pipeline().store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Không có run này")
    return run


def _brief_spans(spans: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if spans is None:
        return None
    return [{"level": s["level"], "text": s["text"], "start": s["start"], "end": s["end"]} for s in spans]


def _case_summary(row: dict[str, Any]) -> dict[str, Any]:
    record = json.loads(row["record"])
    layers = record.get("layers", {})
    judge = layers.get("judge", {})
    error = next((v for k, v in layers.items() if k.endswith("_error")), None)
    return {
        "case_id": row["case_id"], "position": row["position"], "stage": row["stage"],
        "decision": row["decision"], "cost_usd": row["cost_usd"], "text": record["raw_text"],
        "audit": record["meta"].get("audit", False),
        "model": _brief_spans(record["old"]["spans"]),
        "deepseek": _brief_spans((record.get("new") or {}).get("spans")),
        "winners": judge.get("winners"),
        "severity": record.get("diff", {}).get("severity_total"),
        "confirmed_error": layers.get("gate", {}).get("confirmed_error"),
        "model_layer0_ok": layers.get("layer0_model", {}).get("ok"),
        "error": error.get("message") if error else None,
    }


@app.post("/api/runs/preview")
def preview_run(request: UploadRequest):
    try:
        return _pipeline().preview(request.filename, request.content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/runs")
def create_run(request: RunRequest):
    try:
        run_id = _pipeline().submit(request.filename, request.content, request.kind, request.gold)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"run_id": run_id}


@app.get("/api/runs")
def list_runs():
    runs = _pipeline().store.list_runs()
    return [{k: run[k] for k in ("run_id", "filename", "kind", "status", "n_cases", "budget_usd", "cost_usd",
                                 "created_at", "finished_at")} for run in runs]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = _run_or_404(run_id)
    return {**run, "progress": _pipeline().progress(run_id)}


@app.get("/api/runs/{run_id}/cases")
def list_cases(run_id: str):
    _run_or_404(run_id)
    return [_case_summary(row) for row in _pipeline().store.case_rows(run_id)]


@app.get("/api/runs/{run_id}/cases/{case_id:path}")
def get_case(run_id: str, case_id: str):
    _run_or_404(run_id)
    rows = _pipeline().store.broker.query("SELECT record FROM cases WHERE run_id = ? AND case_id = ?",
                                          (run_id, case_id))
    if not rows:
        raise HTTPException(status_code=404, detail="Không có case này")
    return json.loads(rows[0]["record"])


@app.get("/api/pipeline/config")
def pipeline_config():
    from feedback.core.config import load_yaml
    from feedback.stages import qwen_judge

    thresholds = load_yaml("thresholds")
    try:
        fingerprint, parts = qwen_judge.fingerprint()
    except Exception as exc:                                  # thiếu key trong .env
        return {"error": str(exc), "thresholds": thresholds}
    calibration = _pipeline().store.latest_calibration(fingerprint)
    return {"thresholds": thresholds, "judge_fingerprint": fingerprint, "judge_config": parts,
            "calibrated": calibration is not None,
            "calibration": {"run_id": calibration["run_id"], "created_at": calibration["created_at"]}
            if calibration else None}


@app.get("/api/runs/{run_id}/label-queue")
def label_queue(run_id: str):
    """Hàng chờ gán nhãn tay (random audit + judge lật), chỉ có text — xem feedback/eval/make_gold_sample.py."""
    from feedback.eval.make_gold_sample import to_label_file

    _run_or_404(run_id)
    return to_label_file(run_id, _pipeline().store.records(run_id))
