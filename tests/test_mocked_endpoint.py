from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from pm_coder import async_run_auto, run_auto_sync


class MockOpenAIHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []
    lock = Lock()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        with self.lock:
            self.requests.append(payload)
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": payload.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "mock response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 4,
                "total_tokens": 15,
            },
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args: object) -> None:
        return


class MockOpenAIServer:
    def __enter__(self) -> MockOpenAIServer:
        MockOpenAIHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"


def _common_options(server: MockOpenAIServer, tmp_path: Path) -> dict[str, Any]:
    return {
        "cwd": tmp_path,
        "log_root": tmp_path / "logs",
        "base_url": server.base_url,
        "api_key": "test",
        "model": "mock-model",
        "shell": "powershell" if os.name == "nt" else "bash",
    }


def test_auto_mode_uses_mocked_openai_endpoint_and_continues_history(
    tmp_path: Path,
) -> None:
    with MockOpenAIServer() as server:
        options = _common_options(server, tmp_path)
        first = asyncio.run(async_run_auto("first", **options))
        second = asyncio.run(async_run_auto("second", run_id=first.run_id, **options))

        assert first.response == "mock response"
        assert second.run_id == first.run_id
        assert second.tokens_used["total_tokens"] == 15
        assert len(MockOpenAIHandler.requests) == 2
        assert len(MockOpenAIHandler.requests[1]["messages"]) > len(
            MockOpenAIHandler.requests[0]["messages"]
        )
        assert "max_tokens" not in MockOpenAIHandler.requests[0]
        assert (tmp_path / "logs" / first.run_id / "messages.json").is_file()


def test_sync_accessor_works_in_a_regular_thread(tmp_path: Path) -> None:
    with MockOpenAIServer() as server:
        options = _common_options(server, tmp_path)
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(run_auto_sync, "thread prompt", **options).result()

        assert result["response"] == "mock response"
        assert result["run_id"]
        assert result["tokens_used"]["output_tokens"] == 4
