"""Adapters for the two local PhoBERT models and the paid OpenRouter model."""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from scripts.layer1_adapter import LABELS, normalize


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
PROMPT_PATH = ROOT / "prompt/v2/compiled/system_prompt_v2_with_partial_input.txt"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
REASONING_BUDGETS = (10000, 5000, 3000)

_tokenizer = None
_torch_model = None
_onnx_session = None
_load_locks = {"tokenizer": threading.Lock(), "pytorch_fp32": threading.Lock(), "onnx_fp32": threading.Lock()}
_inference_locks = {"pytorch_fp32": threading.Lock(), "onnx_fp32": threading.Lock()}
_qwen_lock = threading.Lock()


def loaded_status():
    return {"pytorch_fp32": _torch_model is not None, "onnx_fp32": _onnx_session is not None}


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with _load_locks["tokenizer"]:
            if _tokenizer is None:
                from transformers import AutoTokenizer

                _tokenizer = AutoTokenizer.from_pretrained(
                    str(MODEL_DIR / "phobert-ner-address-onnx"), local_files_only=True
                )
    return _tokenizer


def _get_torch_model():
    global _torch_model
    if _torch_model is None:
        with _load_locks["pytorch_fp32"]:
            if _torch_model is None:
                from transformers import AutoModelForTokenClassification

                _torch_model = AutoModelForTokenClassification.from_pretrained(
                    str(MODEL_DIR / "phobert-ner-address-pytorch"), local_files_only=True
                ).eval()
    return _torch_model


def _get_onnx_session():
    global _onnx_session
    if _onnx_session is None:
        with _load_locks["onnx_fp32"]:
            if _onnx_session is None:
                import onnxruntime as ort

                files = sorted((MODEL_DIR / "phobert-ner-address-onnx").glob("*.onnx"))
                if not files:
                    raise FileNotFoundError("Không tìm thấy model ONNX FP32")
                options = ort.SessionOptions()
                options.intra_op_num_threads = 4
                _onnx_session = ort.InferenceSession(
                    str(files[0]), options, providers=["CPUExecutionProvider"]
                )
    return _onnx_session


def _base_result(engine):
    return {
        "engine": engine, "ok": False, "error": None, "spans": [], "level_chain": "",
        "raw_labels": None, "elapsed_sec": 0.0, "cost_usd": 0.0, "meta": {},
    }


def _local_parse(text, engine):
    started = time.monotonic()
    result = _base_result(engine)
    try:
        tokenizer = _get_tokenizer()
        with _inference_locks[engine]:
            if engine == "pytorch_fp32":
                import torch

                model = _get_torch_model()
                encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                with torch.inference_mode():
                    ids = model(**encoded).logits.argmax(-1)[0].tolist()
                token_ids = encoded["input_ids"][0].tolist()
            else:
                session = _get_onnx_session()
                encoded = tokenizer(text, return_tensors="np", truncation=True, max_length=128)
                input_names = {entry.name for entry in session.get_inputs()}
                feeds = {key: value.astype(np.int64) for key, value in encoded.items() if key in input_names}
                ids = session.run(None, feeds)[0].argmax(-1)[0].tolist()
                token_ids = encoded["input_ids"][0].tolist()

        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        pairs = [[token, LABELS[label]] for token, label in zip(tokens, ids)
                 if token not in ("<s>", "</s>")]
        spans = [{"level": level, "text": value, "start": None, "end": None,
                  "truncated": None} for level, value in normalize(pairs)]
        result.update(ok=True, spans=spans,
                      level_chain=">".join(span["level"] for span in spans),
                      raw_labels=[{"token": token, "label": label} for token, label in pairs],
                      meta={"token_count": len(pairs), "input_truncated": len(token_ids) >= 128})
    except Exception as exc:
        result["error"] = str(exc)
    result["elapsed_sec"] = round(time.monotonic() - started, 3)
    return result


def parse_pytorch(text):
    return _local_parse(text, "pytorch_fp32")


def parse_onnx(text):
    return _local_parse(text, "onnx_fp32")


def _config():
    config = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                config[key] = value.strip().strip('"').strip("'")
    return config


def qwen_configured():
    config = _config()
    return bool(config.get("OPENROUTER_API") and config.get("OPENROUTER_MODEL") and PROMPT_PATH.exists())


def _extract(raw):
    if not raw:
        return None
    match = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    content = (match.group(1) if match else raw).strip()
    try:
        return json.loads(content)
    except (ValueError, TypeError):
        match = re.search(r"[\{\[].*[\}\]]", content, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except ValueError:
                pass
    return None


def _post_qwen(body, api_key):
    request = urllib.request.Request(
        OPENROUTER_URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def parse_qwen(text):
    if not _qwen_lock.acquire(blocking=False):
        result = _base_result("qwen")
        result["error"] = "Qwen đang xử lý yêu cầu khác; chưa gửi thêm yêu cầu tính phí"
        return result
    try:
        return _parse_qwen_locked(text)
    finally:
        _qwen_lock.release()


def _parse_qwen_locked(text):
    started = time.monotonic()
    result = _base_result("qwen")
    attempts = []
    total_cost = 0.0
    try:
        config = _config()
        if not qwen_configured():
            raise RuntimeError("Chưa cấu hình OPENROUTER_API, OPENROUTER_MODEL hoặc prompt Qwen")
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        use_response_format = True
        for index, budget in enumerate(REASONING_BUDGETS):
            body = {
                "model": config["OPENROUTER_MODEL"], "temperature": 0,
                "max_tokens": 12000, "reasoning": {"max_tokens": budget},
                "usage": {"include": True},
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps({"text": text}, ensure_ascii=False)},
                ],
            }
            if use_response_format:
                body["response_format"] = {"type": "json_object"}
            try:
                response = _post_qwen(body, config["OPENROUTER_API"])
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:400]
                if exc.code == 400 and use_response_format and "response_format" in detail:
                    use_response_format = False
                    body.pop("response_format", None)
                    attempts.append({"attempt": index + 1, "note": "response_format không được hỗ trợ"})
                    try:
                        response = _post_qwen(body, config["OPENROUTER_API"])
                    except urllib.error.HTTPError as second:
                        detail = second.read().decode("utf-8", "replace")[:400]
                        exc = second
                    else:
                        exc = None
                if exc is not None:
                    if exc.code == 403 and "key limit exceeded" in detail.lower():
                        result["error"] = "Key OpenRouter đã hết hạn mức chi tiêu"
                        break
                    attempts.append({"attempt": index + 1, "http_error": exc.code})
                    result["error"] = f"OpenRouter HTTP {exc.code}: {detail[:150]}"
                    if index < len(REASONING_BUDGETS) - 1:
                        time.sleep(4 * (index + 1))
                    continue
            except TimeoutError:
                result["error"] = f"OpenRouter timeout sau {round(time.monotonic() - started, 1)} giây"
                attempts.append({"attempt": index + 1, "timeout": True})
                break
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, TimeoutError) or "timed out" in str(exc.reason).lower():
                    result["error"] = f"OpenRouter timeout sau {round(time.monotonic() - started, 1)} giây"
                else:
                    result["error"] = f"Lỗi kết nối OpenRouter: {str(exc.reason)[:180]}"
                attempts.append({"attempt": index + 1, "error": str(exc.reason)[:180]})
                break
            except Exception as exc:
                result["error"] = f"Lỗi OpenRouter: {str(exc)[:180]}"
                attempts.append({"attempt": index + 1, "error": str(exc)[:180]})
                break

            usage = response.get("usage") or {}
            total_cost += float(usage.get("cost") or 0)
            content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
            parsed = _extract(content)
            attempts.append({
                "attempt": index + 1, "reasoning_budget": budget,
                "cost_usd": float(usage.get("cost") or 0), "empty": parsed is None,
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            })
            if isinstance(parsed, dict) and isinstance(parsed.get("spans"), list):
                spans = []
                for span in parsed["spans"]:
                    if not isinstance(span, dict) or not isinstance(span.get("level"), str) or not isinstance(span.get("text"), str):
                        continue
                    spans.append({key: span.get(key) for key in ("level", "text", "start", "end", "truncated")})
                spans.sort(key=lambda span: (span["start"] is None, span["start"] if isinstance(span["start"], int) else 0))
                result.update(ok=True, error=None, spans=spans,
                              level_chain=">".join(str(span["level"]) for span in spans),
                              meta={"prompt_tokens": usage.get("prompt_tokens"),
                                    "completion_tokens": usage.get("completion_tokens"),
                                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                                    "retries": index, "attempts": attempts})
                break
            result["error"] = (
                f"Qwen trả về rỗng hoặc JSON không có spans sau {index + 1} lần thử; "
                f"đã tốn ${total_cost:.5f}"
            )
            if index < len(REASONING_BUDGETS) - 1:
                time.sleep(2)
    except Exception as exc:
        result["error"] = str(exc)
    result["cost_usd"] = round(total_cost, 8)
    result["elapsed_sec"] = round(time.monotonic() - started, 3)
    result["meta"].setdefault("attempts", attempts)
    return result
