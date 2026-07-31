from __future__ import annotations

import json
from pathlib import Path

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from pm_coder import SessionStore, parse_args, render_messages_yaml


def test_session_store_round_trips_structured_messages(tmp_path: Path) -> None:
    store = SessionStore.open(tmp_path / "project", log_root=tmp_path / "logs")
    messages = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        ModelResponse(parts=[TextPart(content="world")]),
    ]

    store.save_messages(messages)

    loaded = store.load_messages()
    assert len(loaded) == 2
    assert loaded[0].parts[0].content == "hello"
    assert loaded[1].parts[0].content == "world"
    assert json.loads(store.messages_path.read_text(encoding="utf-8"))[0]["parts"][0]["content"] == "hello"


def test_session_store_writes_human_readable_markdown_yaml(tmp_path: Path) -> None:
    store = SessionStore.open(tmp_path / "project", log_root=tmp_path / "logs")
    messages = [
        ModelRequest(parts=[UserPromptPart(content="# Heading\n\n- Keep this markdown")]),
        ModelResponse(parts=[TextPart(content="A response with a preserved\nline break.")]),
    ]

    store.save_messages(messages)

    rendered = store.human_messages_path.read_text(encoding="utf-8")
    assert "content: |" in rendered
    assert "# Heading" in rendered
    assert "- Keep this markdown" in rendered


def test_existing_json_can_be_rendered_without_a_new_turn(tmp_path: Path) -> None:
    store = SessionStore.open(tmp_path / "project", log_root=tmp_path / "logs")
    store.save_messages([ModelRequest(parts=[UserPromptPart(content="existing")])])
    store.human_messages_path.unlink()

    output = render_messages_yaml(store.messages_path)

    assert output == store.human_messages_path
    assert "existing" in output.read_text(encoding="utf-8")


def test_new_session_id_contains_timestamp_and_working_directory(tmp_path: Path) -> None:
    store = SessionStore.open(tmp_path / "my-project", log_root=tmp_path / "logs")

    assert store.run_id.endswith("_my-project")
    assert len(store.run_id.split("_", 2)[0]) == 10
    assert (store.path / "session.json").is_file()


def test_auto_cli_arguments_accept_literal_prompt_and_file_prompt() -> None:
    literal = parse_args(["--auto", "hello"])
    file_prompt = parse_args(["--mode", "auto", "--prompt-file", "prompt.txt", "--run-id", "old"])

    assert literal.mode == "auto"
    assert literal.prompt == "hello"
    assert file_prompt.mode == "auto"
    assert file_prompt.prompt_file == "prompt.txt"
    assert file_prompt.run_id == "old"
    assert literal.max_tokens == 8_192
    assert literal.max_tool_output is None
    assert literal.max_skill_index is None
    assert literal.max_project_instructions is None
