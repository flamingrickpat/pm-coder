from __future__ import annotations

from pathlib import Path

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_core import to_jsonable_python

from pm_coder import (
    _find_cut_index,
    _is_cut_point,
    _merge_summary_into_first_request,
    build_system_prompt,
    discover_workspace,
    estimate_context_tokens,
    estimate_message_tokens,
    format_skill_index,
    parse_args,
    build_settings,
    one_shot_options,
)


def _turn(question: str, answer: str) -> list[ModelRequest | ModelResponse]:
    return [
        ModelRequest(parts=[UserPromptPart(content=question)]),
        ModelResponse(parts=[TextPart(content=answer)]),
    ]


def _as_data(messages: list[object]) -> list[dict]:
    data = to_jsonable_python(messages)
    assert isinstance(data, list)
    return data


def test_estimate_message_tokens_is_bounded_and_positive() -> None:
    data = _as_data([ModelRequest(parts=[UserPromptPart(content="x" * 400)])])
    tokens = estimate_message_tokens(data[0])
    # chars/4 with a floor of 1
    assert tokens == 100


def test_estimate_context_tokens_estimates_everything_when_no_usage() -> None:
    messages: list[object] = []
    for i in range(3):
        messages.extend(_turn(f"q{i} " * 100, f"a{i} " * 200))
    data = _as_data(messages)
    total = sum(estimate_message_tokens(m) for m in data)
    assert estimate_context_tokens(data) == total


def test_cut_points_only_at_fresh_user_turns() -> None:
    messages: list[object] = []
    for i in range(3):
        messages.append(ModelRequest(parts=[UserPromptPart(content=f"q{i}")]))
        messages.append(
            ModelResponse(parts=[ToolCallPart(tool_name="host_shell", args={"command": "x"})])
        )
        messages.append(
            ModelRequest(parts=[ToolReturnPart(tool_name="host_shell", content="out")])
        )
        messages.append(ModelResponse(parts=[TextPart(content=f"a{i}")]))
    data = _as_data(messages)
    cut_points = [i for i, m in enumerate(data) if _is_cut_point(m)]
    # indices 0, 4, 8 are the fresh user turns; tool-return requests (2, 6) excluded
    assert cut_points == [0, 4, 8]


def test_find_cut_index_keeps_recent_context() -> None:
    messages: list[object] = []
    for i in range(30):
        messages.extend(_turn(f"q{i} " * 50, f"a{i} " * 200))
    data = _as_data(messages)
    cut = _find_cut_index(data, keep_recent_tokens=0)
    assert cut is not None
    # with a zero budget we keep only the most recent messages; the cut must be
    # at a fresh user-turn boundary and cut something off
    assert cut > 0
    assert _is_cut_point(data[cut])
    kept = sum(estimate_message_tokens(m) for m in data[cut:])
    prefix = sum(estimate_message_tokens(m) for m in data[:cut])
    assert prefix > 0


def test_merge_summary_into_first_request_prepends_summary() -> None:
    request = ModelRequest(parts=[UserPromptPart(content="first kept user turn")])
    merged = _merge_summary_into_first_request(request, "THE_SUMMARY")
    assert isinstance(merged, ModelRequest)
    parts = list(merged.parts)
    assert isinstance(parts[0], UserPromptPart)
    assert "THE_SUMMARY" in str(parts[0].content)
    assert "first kept user turn" in str(parts[0].content)


def test_skill_selection_and_injection(tmp_path: Path) -> None:
    skill_dir = tmp_path / ".agents" / "skills" / "deploy"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: deploy\n---\n# Deploy\nRun the pipeline and push.\n",
        encoding="utf-8",
    )
    args = parse_args(["--cwd", str(tmp_path), "--skill", "deploy"])
    settings = build_settings(args)
    discovery = discover_workspace(settings)
    assert discovery.selected_skill is not None
    assert discovery.selected_skill.name == "deploy"
    prompt = build_system_prompt(
        settings,
        format_skill_index([], None),
        "<project_instructions />",
        discovery.mcp_server_names,
        discovery.selected_skill,
    )
    assert '<selected_skill name="deploy">' in prompt
    assert "Run the pipeline and push." in prompt


def test_unknown_skill_raises(tmp_path: Path) -> None:
    import pytest

    args = parse_args(["--cwd", str(tmp_path), "--skill", "does-not-exist"])
    settings = build_settings(args)
    with pytest.raises(FileNotFoundError):
        discover_workspace(settings)


def test_rate_limit_defaults_to_zero_unlimited() -> None:
    # 0 means unlimited, so a plain invocation should be unlimited by default.
    args = parse_args([])
    assert args.request_limit == 0
    assert args.wall_clock_limit == 0
    # and a default (0) must not accidentally satisfy legacy one-shot validation
    assert one_shot_options(args) is None


def test_nonzero_rate_limits_passed_through(tmp_path: Path) -> None:
    args = parse_args(["--request-limit", "30", "--wall-clock-limit", "300"])
    assert args.request_limit == 30
    assert args.wall_clock_limit == 300
