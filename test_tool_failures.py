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


def test_subagent_returns_a_failure_result_for_invalid_arguments(tmp_path: Path) -> None:
    subagent = make_subagent_tool(_settings(tmp_path)).function

    result = _failure(subagent("", None))

    assert "has no len" in result["error"]
