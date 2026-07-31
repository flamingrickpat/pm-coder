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
    finish_reasons: list[str] = []
    failures: list[tuple[int, str]] = []
    lock = Lock()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        with self.lock:
            self.requests.append(payload)
            failure = self.failures.pop(0) if self.failures else None
            finish_reason = (
                self.finish_reasons.pop(0) if self.finish_reasons else "stop"
            )
        if failure is not None:
            status, message = failure
            self._send_json(
                status,
                {"error": {"message": message, "type": "server_error"}},
            )
            return
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": payload.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            "discarded incomplete response"
                            if finish_reason == "length"
                            else "mock response"
                        ),
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 8_192 if finish_reason == "length" else 4,
                "total_tokens": 8_203 if finish_reason == "length" else 15,
            },
        }
        self._send_json(200, response)

    def _send_json(self, status: int, value: dict[str, Any]) -> None:
        encoded = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args: object) -> None:
        return


class MockOpenAIServer:
    def __enter__(self) -> MockOpenAIServer:
        MockOpenAIHandler.requests = []
        MockOpenAIHandler.finish_reasons = []
        MockOpenAIHandler.failures = []
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
        assert MockOpenAIHandler.requests[0]["max_completion_tokens"] == 8_192
        assert (tmp_path / "logs" / first.run_id / "messages.json").is_file()


def test_sync_accessor_works_in_a_regular_thread(tmp_path: Path) -> None:
    with MockOpenAIServer() as server:
        options = _common_options(server, tmp_path)
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(run_auto_sync, "thread prompt", **options).result()

        assert result["response"] == "mock response"
        assert result["run_id"]
        assert result["tokens_used"]["output_tokens"] == 4


def test_auto_mode_recovers_from_a_response_limit_with_shorter_call_feedback(
    tmp_path: Path,
) -> None:
    with MockOpenAIServer() as server:
        MockOpenAIHandler.finish_reasons = ["length", "stop"]
        result = asyncio.run(async_run_auto("finish the role", **_common_options(server, tmp_path)))

        assert result.response == "mock response"
        assert len(MockOpenAIHandler.requests) == 2
        retry_messages = json.dumps(
            MockOpenAIHandler.requests[1]["messages"],
            ensure_ascii=False,
        )
        assert "reached 8,192 generated tokens" in retry_messages
        assert "below about 4,000 generated tokens" in retry_messages
        assert "discarded incomplete response" not in retry_messages
        assert result.tokens_used["requests"] == 2


def test_auto_mode_explains_and_recovers_from_malformed_tool_call_json(
    tmp_path: Path,
) -> None:
    with MockOpenAIServer() as server:
        error = (
            "Failed to parse tool call arguments as JSON: "
            "missing closing quote at column 8192"
        )
        # The OpenAI client retries twice before pm-coder sees the HTTP 500.
        MockOpenAIHandler.failures = [(500, error)] * 3
        result = asyncio.run(async_run_auto("finish the role", **_common_options(server, tmp_path)))

        assert result.response == "mock response"
        assert len(MockOpenAIHandler.requests) == 4
        retry_messages = json.dumps(MockOpenAIHandler.requests[-1]["messages"])
        assert "because its JSON was incomplete" in retry_messages
        assert "the tool did not run" in retry_messages


def test_auto_mode_retries_a_server_restart_without_spending_agent_requests(
    tmp_path: Path,
) -> None:
    with MockOpenAIServer() as server:
        # The provider's two retries and pm-coder's retry all represent one
        # logical agent request. None reached the model.
        MockOpenAIHandler.failures = [(503, "server is restarting")] * 3
        result = asyncio.run(async_run_auto(
            "finish the role",
            request_limit=1,
            **_common_options(server, tmp_path),
        ))

        assert result.response == "mock response"
        assert len(MockOpenAIHandler.requests) == 4
        retry_messages = json.dumps(MockOpenAIHandler.requests[-1]["messages"])
        assert "finish the role" in retry_messages
        assert "endpoint disconnected or restarted" not in retry_messages
        assert result.tokens_used["requests"] == 1
