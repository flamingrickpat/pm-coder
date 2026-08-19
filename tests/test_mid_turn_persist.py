from __future__ import annotations

import asyncio
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

from pm_coder import async_run_auto


class _ToolCallOpenAIHandler(BaseHTTPRequestHandler):
    """Serves a shell tool call first, then a plain "done" response."""

    calls = 0
    tool_name: str = "bash"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        type(self).calls += 1
        model = payload.get("model", "mock-model")
        if type(self).calls == 1:
            sleep_arg = "Start-Sleep -Seconds 2" if os.name == "nt" else "sleep 2"
            response = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": type(self).tool_name,
                                        "arguments": json.dumps(
                                            {"command": sleep_arg}
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        else:
            response = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "done",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                },
            }
        self._send(200, response)

    def _send(self, status: int, value: dict[str, Any]) -> None:
        encoded = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args: object) -> None:
        return


class _ToolCallOpenAIServer:
    def __enter__(self) -> _ToolCallOpenAIServer:
        _ToolCallOpenAIHandler.calls = 0
        shell = "powershell" if os.name == "nt" else "bash"
        _ToolCallOpenAIHandler.tool_name = shell
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _ToolCallOpenAIHandler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


def test_session_file_is_flushed_during_a_tool_call(tmp_path: Path) -> None:
    """The messages file must appear mid-turn, before the tool finishes."""
    with _ToolCallOpenAIServer() as server:
        shell = "powershell" if os.name == "nt" else "bash"
        sleep_arg = "Start-Sleep -Seconds 2" if os.name == "nt" else "sleep 2"
        log_root = tmp_path / "logs"

        async def go() -> None:
            base_url = (
                f"http://127.0.0.1:{server.server.server_address[1]}/v1"
            )
            task = asyncio.create_task(
                async_run_auto(
                    "inspect the workspace",
                    cwd=str(tmp_path),
                    run_id="midturn",
                    log_root=str(log_root),
                    base_url=base_url,
                    api_key="test",
                    model="mock-model",
                    shell=shell,
                )
            )
            # The tool sleeps for ~2s after the before-flush writes the file,
            # so we have a window to observe the mid-turn write from in here.
            messages_json = log_root / "midturn" / "messages.json"
            deadline = time.monotonic() + 6
            found = False
            text = ""
            while time.monotonic() < deadline:
                if messages_json.exists():
                    text = messages_json.read_text(encoding="utf-8")
                    if sleep_arg in text:
                        found = True
                        break
                await asyncio.sleep(0.03)
            result = await asyncio.wait_for(task, timeout=15)
            assert result.response == "done"
            assert found, (
                "session file was not flushed mid-turn while the tool ran; "
                f"file content: {text!r}"
            )

        asyncio.run(go())
