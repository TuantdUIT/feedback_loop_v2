"""Đọc capture curl và ghép từng text với đúng kết quả cùng chỉ số."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse


def parse_capture(path: str | Path) -> list[dict]:
    """Bóc texts[] và results[] từ một capture; từ chối lô lệch độ dài."""
    source = Path(path)
    content = source.read_text(encoding="utf-8")
    request_match = re.search(r"--data\s*'", content)
    if request_match is None:
        raise ValueError(f"{source}: không tìm thấy --data trong capture")
    decoder = json.JSONDecoder()
    try:
        request, _ = decoder.raw_decode(content[request_match.end():].lstrip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: request JSON không hợp lệ") from exc

    response_match = re.search(r'(?m)^\s*\{\s*"results"\s*:', content)
    if response_match is None:
        raise ValueError(f"{source}: không tìm thấy response results[]")
    response_start = content.index("{", response_match.start())
    try:
        response, _ = decoder.raw_decode(content[response_start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source}: response JSON không hợp lệ") from exc

    texts = request.get("texts") if isinstance(request, dict) else None
    results = response.get("results") if isinstance(response, dict) else None
    if not isinstance(texts, list) or not isinstance(results, list):
        raise ValueError(f"{source}: texts và results phải là list")
    if len(texts) != len(results):
        raise ValueError(f"{source}: texts ({len(texts)}) khác results ({len(results)})")

    url_match = re.search(r"curl\s+--location\s+'([^']+)'", content)
    if url_match is None:
        raise ValueError(f"{source}: không tìm thấy URL endpoint trong capture")
    endpoint = urlparse(url_match.group(1)).path
    payloads = []
    for index, (text, item) in enumerate(zip(texts, results, strict=True)):
        if not isinstance(item, dict):
            raise ValueError(f"{source}: results[{index}] không phải object")
        payloads.append({
            "text": text,
            "result": item.get("result"),
            "source": {"capture": source.name, "endpoint": endpoint, "batch_index": index},
        })
    return payloads
