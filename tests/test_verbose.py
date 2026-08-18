from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pm_coder import (
    _VerbosePrinter,
    VerboseModel,
    VerboseStreamedResponse,
    _maybe_compact_history,
    async_run_auto,
    build_agent,
    build_settings,
    discover_workspace,
    parse_args,
)
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters, StreamedResponse

from test_mocked_endpoint import MockOpenAIServer


def test_parse_args_verbose_flag() -> None:
    assert parse_args(["--cwd", "."]).verbose is False
    assert parse_args(["--cwd", ".", "-v"]).verbose is True
    assert parse_args(["--cwd", ".", "--verbose"]).verbose is True


def test_build_settings_propagates_verbose(tmp_path: Path) -> None:
    settings = build_settings(parse_args(["--cwd", str(tmp_path), "-v"]))
    assert settings.verbose is True
    plain = build_settings(parse_args(["--cwd", str(tmp_path)]))
    assert plain.verbose is False


def test_verbose_printer_streams_raw_text() -> None:
    out = StringIO()
    printer = _VerbosePrinter(out, label="agent")
    printer.request_banner("qwen", 1, 1)
    printer.handle(PartStartEvent(index=0, part=TextPart(content="hel")))
    printer.handle(
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="lo ")),
    )
    printer.handle(
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="world")),
    )
    printer.handle(
        PartEndEvent(
            index=0,
            part=TextPart(content="hello world"),
        ),
    )
    printer.response_done(
        SimpleNamespace(
            finish_reason="stop",
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        )
    )
    dumped = out.getvalue()
    assert "pm-coder verbose agent: request: model=qwen messages=1 tools=1" in dumped
    assert "assistant text:" in dumped
    assert "hello world" in dumped
    assert "finish_reason=stop" in dumped
    assert "input=5" in dumped
    assert "output=3" in dumped


def test_verbose_printer_streams_tool_call_args() -> None:
    out = StringIO()
    printer = _VerbosePrinter(out, label="agent")
    printer.handle(
        PartStartEvent(
            index=0,
            part=ToolCallPart(
                tool_name="powershell", args="", tool_call_id="call-1"
            ),
        ),
    )
    printer.handle(
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(args_delta='{"command": "'),
        ),
    )
    printer.handle(
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(args_delta='Get-ChildItem"}'),
        ),
    )
    printer.handle(
        PartEndEvent(
            index=0,
            part=ToolCallPart(
                tool_name="powershell",
                args='{"command": "Get-ChildItem"}',
                tool_call_id="call-1",
            ),
        ),
    )
    dumped = out.getvalue()
    assert "tool call: powershell" in dumped
    # The raw JSON stream must land verbatim on the verbose stream.
    assert '{"command": "Get-ChildItem"}' in dumped


class _FakeInnerStream(StreamedResponse):
    """Minimal StreamedResponse that streams two text deltas."""

    def __init__(self) -> None:
        super().__init__(ModelRequestParameters())

    async def _get_event_iterator(self) -> Any:
        for event in self._parts_manager.handle_text_delta(
            vendor_part_id=None, content="hello "
        ):
            yield event
        for event in self._parts_manager.handle_text_delta(
            vendor_part_id=None, content="world"
        ):
            yield event

    async def close_stream(self) -> None:
        return None

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def provider_name(self) -> str | None:
        return None

    @property
    def provider_url(self) -> str | None:
        return None

    @property
    def timestamp(self) -> datetime:
        return datetime.now(UTC)


def test_verbose_streamed_response_tees_events() -> None:
    out = StringIO()
    inner = _FakeInnerStream()
    wrapper = VerboseStreamedResponse(inner, _VerbosePrinter(out, label="test"))
    seen: list[Any] = []

    async def collect() -> None:
        async for event in wrapper:
            seen.append(event)

    asyncio.run(collect())

    # Every provider event passes through untouched (a text part start in
    # text-output mode also emits the framework's final_result event). The
    # first delta call creates the part (start event, no delta); the second
    # one appends content (one part_delta).
    kinds = [event.event_kind for event in seen]
    assert kinds == [
        "part_start",
        "final_result",
        "part_delta",
        "part_end",
    ]
    assert wrapper.final_result_event is not None
    # Metadata and state delegate to the inner stream. (finish_reason is
    # not asserted: the fake emits no completion-parameters update, so the
    # synthesized response keeps it unset.)
    assert wrapper.model_name == "fake-model"
    assert "hello world" in str(wrapper.get().parts[0].content)
    dumped = out.getvalue()
    assert "hello world" in dumped
    assert "response complete" in dumped


def _mock_options(server: MockOpenAIServer, tmp_path: Path) -> dict[str, Any]:
    return {
        "cwd": tmp_path,
        "log_root": tmp_path / "logs",
        "base_url": server.base_url,
        "api_key": "test",
        "model": "mock-model",
        "shell": "powershell" if os.name == "nt" else "bash",
    }


def test_verbose_streams_raw_model_output_to_stdout(
    tmp_path: Path,
    capsys: Any,
) -> None:
    with MockOpenAIServer() as server:
        options = _mock_options(server, tmp_path)
        result = asyncio.run(async_run_auto("first", verbose=True, **options))

    assert result.response == "mock response"
    dumped = capsys.readouterr().out
    assert "pm-coder verbose agent: request: model=mock-model" in dumped
    assert "messages=1 tools=2" in dumped
    # The raw generated text must be visible on stdout even though this is
    # the very first message of the session.
    assert "mock response" in dumped
    assert "response complete: finish_reason=stop" in dumped


def test_non_verbose_auto_mode_stays_quiet_on_stdout(
    tmp_path: Path,
    capsys: Any,
) -> None:
    with MockOpenAIServer() as server:
        options = _mock_options(server, tmp_path)
        result = asyncio.run(async_run_auto("first", **options))

    assert result.response == "mock response"
    dumped = capsys.readouterr().out
    assert "pm-coder verbose" not in dumped


def test_verbose_prints_auto_compact_summary(
    tmp_path: Path,
    capsys: Any,
) -> None:
    with MockOpenAIServer() as server:
        args = parse_args(
            [
                "--cwd",
                str(tmp_path),
                "--base-url",
                server.base_url,
                "--api-key",
                "test",
                "--model",
                "mock-model",
                "--shell",
                "powershell" if os.name == "nt" else "bash",
                "--auto-compact",
                "--context-window",
                "10",
                "--compact-keep-recent-tokens",
                "4",
                "-v",
            ]
        )
        settings = build_settings(args)
        agent = build_agent(settings, discover_workspace(settings))
        history = [
            ModelRequest(parts=[UserPromptPart(content="old question")]),
            ModelResponse(parts=[TextPart(content="old answer")]),
            ModelRequest(parts=[UserPromptPart(content="new question")]),
            ModelResponse(parts=[TextPart(content="new answer")]),
        ]

        async def go() -> Any:
            async with agent:
                return await _maybe_compact_history(agent, settings, history)

        new_history, summary_requests = asyncio.run(go())

    assert summary_requests == 1
    assert len(new_history) == 2
    merged = new_history[0].parts[0].content
    assert "mock response" in merged
    assert "new question" in merged

    dumped = capsys.readouterr().out
    assert "pm-coder verbose compact: context" in dumped
    assert "exceeds threshold" in dumped
    # The summarization pass itself streams through the verbose model.
    assert "pm-coder verbose compact: request: model=mock-model" in dumped
    assert "pm-coder verbose compact: summary:\nmock response" in dumped


def test_verbose_model_wraps_and_delegates() -> None:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    inner = OpenAIChatModel(
        "mock-model",
        provider=OpenAIProvider(base_url="http://127.0.0.1:9/v1", api_key="k"),
    )
    model = VerboseModel(inner, stream=StringIO(), label="agent")
    assert model.wrapped is inner
    assert model.model_name == "mock-model"
