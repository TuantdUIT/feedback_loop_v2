"""Mock server chung cho DeepSeek và OpenRouter: test gọi API thật qua HTTP mà không tốn tiền."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import pytest

Handler = Callable[[str, dict[str, Any]], tuple[int, dict[str, Any]]]


class MockAPI:
    """``handler(provider, body) -> (status, json)``; provider là 'deepseek' hoặc 'qwen'."""

    def __init__(self) -> None:
        self.handler: Handler = lambda provider, body: (500, {"error": "chưa đặt handler"})
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.lock = threading.Lock()
        mock = self

        class _Req(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                provider = "qwen" if self.path.startswith("/openrouter") else "deepseek"
                with mock.lock:
                    mock.calls.append((provider, body))
                status, payload = mock.handler(provider, body)
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Req)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def calls_for(self, provider: str) -> list[dict[str, Any]]:
        with self.lock:
            return [body for p, body in self.calls if p == provider]


def chat_response(content: Any, usage: dict[str, Any] | None = None, finish: str = "stop") -> dict[str, Any]:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 1000, "prompt_cache_hit_tokens": 900,
                               "prompt_cache_miss_tokens": 100, "completion_tokens": 50}}


@pytest.fixture()
def mock_api(monkeypatch: pytest.MonkeyPatch) -> MockAPI:
    api = MockAPI()
    monkeypatch.setenv("DEEPSEEK_BASE_URL", api.url + "/deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    monkeypatch.setenv("OPENROUTER_BASE_URL", api.url + "/openrouter")
    monkeypatch.setenv("OPENROUTER_API", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "qwen/test")
    from feedback.core import llm_client
    monkeypatch.setattr(llm_client, "_RESPONSE_FORMAT_OK", {})
    yield api
    api.server.shutdown()
