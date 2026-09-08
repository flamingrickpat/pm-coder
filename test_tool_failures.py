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


def test_read_attaches_a_real_image(tmp_path: Path) -> None:
    """read() on a PNG returns the bytes as a BinaryContent attachment."""
    import base64

    from pydantic_ai.messages import BinaryContent

    payload = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9Q"
        "DwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    (tmp_path / "shot.png").write_bytes(payload)
    read = _function(make_file_tools(_settings(tmp_path)), "read")

    image, note = read("shot.png")

    assert isinstance(image, BinaryContent)
    assert image.data == payload
    assert image.media_type == "image/png"
    assert "image attached" in note


def test_subagent_returns_a_failure_result_for_invalid_arguments(tmp_path: Path) -> None:
    subagent = make_subagent_tool(_settings(tmp_path)).function

    result = _failure(subagent("", None))

    assert "has no len" in result["error"]


def test_file_tools_run_identically_against_a_bash_machine(tmp_path):
    """Same factory, same signatures, same behavior -- only storage switches."""
    from pm_bash_machine import BashMachine

    vm = BashMachine()
    tools = make_file_tools(_settings(tmp_path), vm)
    read = _function(tools, "read")
    write = _function(tools, "write")
    edit = _function(tools, "edit")

    assert "whole file" in write("/home/user/notes.txt", "alpha\nbeta\ngamma\n")
    window = read("/home/user/notes.txt", start_line=2, line_length=1)
    assert "2: beta" in window

    assert edit("/home/user/notes.txt", "beta", "BETA").startswith("edited")
    assert vm.read_text("/home/user/notes.txt") == "alpha\nBETA\ngamma\n"

    # Ranged writes work too: content="" with a valid range deletes the lines.
    assert "lines 2-2" in write("/home/user/notes.txt", "", 2, 2)
    assert vm.read_text("/home/user/notes.txt") == "alpha\ngamma\n"

    result = _failure(read("missing.txt"))
    assert "file does not exist" in result["error"]


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
