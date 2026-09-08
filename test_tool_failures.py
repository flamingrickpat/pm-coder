from __future__ import annotations

import json
from pathlib import Path

from pm_coder import build_settings, make_file_tools, make_subagent_tool


def _settings(tmp_path: Path):
    return build_settings(cwd=tmp_path, model="test", context_window=1024)


def _function(tools, name: str):
    return next(tool.function for tool in tools if tool.name == name)


def _failure(result: str) -> dict[str, object]:
    payload = json.loads(result)
    assert payload["success"] is False
    assert isinstance(payload["error"], str)
    return payload


def test_read_returns_a_failure_result_for_missing_file(tmp_path: Path) -> None:
    read = _function(make_file_tools(_settings(tmp_path)), "read")

    result = _failure(read("missing.txt"))

    assert "file does not exist" in result["error"]


def test_write_returns_a_failure_result_for_invalid_range(tmp_path: Path) -> None:
    target = tmp_path / "document.txt"
    target.write_text("first\nsecond\n", encoding="utf-8")
    write = _function(make_file_tools(_settings(tmp_path)), "write")

    result = _failure(write("document.txt", "replacement", 2, 0))

    assert "invalid write range" in result["error"]
    assert target.read_text(encoding="utf-8") == "first\nsecond\n"


def test_unsupported_image_input_strips_images_and_continues(tmp_path, monkeypatch):
    """llama.cpp without --mmproj answers 500 'image input is not supported'.

    The image sits in the history, so a reconnect-retry resubmits it forever.
    run_turn must instead strip every image, disable read_image, and finish.
    """
    import asyncio
    import base64

    import pm_coder
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import ModelHTTPError
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    png = tmp_path / "shot.png"
    png.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9Q"
        "DwADhgGAWjR9awAAAABJRU5ErkJggg=="
    ))
    settings = pm_coder.build_settings(cwd=tmp_path, model="test", context_window=96_000)
    session = pm_coder.SessionStore(tmp_path / "session", tmp_path)
    session.context_window = 96_000
    monkeypatch.setattr(pm_coder, "active_session", session)

    calls = 0

    def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name="read_image", args={"path": "shot.png"})])
        if pm_coder.count_images(list(messages)):
            raise ModelHTTPError(status_code=500, model_name="test", body={
                "error": {
                    "code": 500,
                    "message": "image input is not supported - hint: if this is "
                               "unexpected, you may need to provide the mmproj",
                    "type": "server_error",
                }
            })
        return ModelResponse(parts=[TextPart(content="done without the image")])

    async def run():
        agent = Agent(FunctionModel(respond), tools=pm_coder.make_file_tools(settings))
        async with agent:
            return await pm_coder.run_turn(agent, settings, session, "check shot.png")

    result = asyncio.run(run())

    assert result.response == "done without the image"
    assert calls == 3
    assert session.vision_supported is False
    assert pm_coder.count_images(session.load_messages()) == 0

    # read_image now declines instead of attaching another image the endpoint
    # would reject again.
    read_image = _function(make_file_tools(_settings(tmp_path)), "read_image")
    assert "does not support image input" in read_image("shot.png")[0]


def test_subagent_returns_a_failure_result_for_invalid_arguments(tmp_path: Path) -> None:
    subagent = make_subagent_tool(_settings(tmp_path)).function

    result = _failure(subagent("", None))

    assert "has no len" in result["error"]


def test_plain_skill_read_includes_content_beyond_default_window(tmp_path):
    skill = tmp_path / "SKILL.md"
    skill.write_text("line\n" * 1100 + "FINAL_SKILL_RULE\n", encoding="utf-8")
    read = _function(make_file_tools(_settings(tmp_path)), "read")
    assert "FINAL_SKILL_RULE" in read(str(skill))
    assert "FINAL_SKILL_RULE" not in read(str(skill), start_line=1, line_length=10)


def setup_module():
    import tempfile
    from pathlib import Path
    import pm_coder

    root = Path(tempfile.mkdtemp(prefix="pm-coder-check-"))
    pm_coder.active_session = pm_coder.SessionStore.open(root, log_root=root / "sessions")
    pm_coder.active_session.context_window = 96_000
