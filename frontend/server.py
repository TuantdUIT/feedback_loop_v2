"""Local comparison UI. Start with python -m uvicorn frontend.server:app --port 8000."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from frontend import engines


STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="So sánh parser địa chỉ")
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
