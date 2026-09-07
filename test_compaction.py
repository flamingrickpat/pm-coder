from pydantic_ai.messages import ToolCallPart

from pm_coder import (
    ModelRequest, ModelResponse, UserPromptPart, ToolReturnPart,
    TextPart, compaction_tail, serialize_for_summary, COMPACT_TAIL_CHARS,
)


def exchange(call_id, content):
    return [
        ModelResponse(parts=[ToolCallPart(tool_name="read", args={"path": "file"}, tool_call_id=call_id)]),
        ModelRequest(parts=[ToolReturnPart(tool_name="read", content=content, tool_call_id=call_id)]),
    ]


def test_oversized_result_is_omitted_without_orphaning_its_call():
    history = [ModelRequest(parts=[UserPromptPart(content="Implement the task")])]
    history += exchange("huge", "x" * 160_000)
    history += exchange("recent", "useful current state")
    tail = compaction_tail(history, 0)
    assert len(tail) == 2
    assert tail[0].parts[0].tool_call_id == "recent"
    assert tail[1].parts[0].tool_call_id == "recent"
    assert len(serialize_for_summary(tail)) <= COMPACT_TAIL_CHARS


def test_checkpoint_is_not_carried_into_the_next_tail():
    history = [ModelRequest(parts=[UserPromptPart(content="task" * 10_000)])]
    history += [ModelResponse(parts=[TextPart(content="[AUTOCOMPACTED EXECUTION CHECKPOINT] old state")])]
    history += exchange("recent", "current state")
    tail = compaction_tail(history, 0)
    assert len(tail) == 2
    assert "AUTOCOMPACTED" not in serialize_for_summary(tail)


def test_oversized_last_exchange_can_leave_an_empty_tail():
    history = [ModelRequest(parts=[UserPromptPart(content="task")])]
    history += exchange("huge", "x" * 160_000)
    assert compaction_tail(history, 0) == []


def test_repeated_recovery_preserves_the_same_recent_exchange():
    history = [ModelRequest(parts=[UserPromptPart(content="task" * 10_000)])]
    history += exchange("recent", "x" * 2_000)
    assert compaction_tail(history, 0)
    assert compaction_tail(history, 16) == compaction_tail(history, 0)


def test_split_summaries_are_merged_even_below_character_budget(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    import pm_coder

    prompts = []

    class SummaryAgent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def run(self, prompt, **kwargs):
            prompts.append(prompt)
            if len(prompts) == 1:
                raise RuntimeError("context window exceeded")
            outputs = {2: "Earlier: read the specification.",
                       3: "Later: specification read; write tests.",
                       4: "Specification read; next write tests."}
            return SimpleNamespace(output=outputs[len(prompts)])

    monkeypatch.setattr(pm_coder, "build_summary_agent", lambda settings: SummaryAgent())
    result = asyncio.run(pm_coder.summarize_text(None, "history " * 10000))
    assert result == "Specification read; next write tests."
    assert len(prompts) == 4
    assert "Earlier: read the specification." in prompts[-1]
    assert "Later: specification read; write tests." in prompts[-1]


def test_selected_external_skill_survives_compaction_and_new_agent(tmp_path, monkeypatch):
    import asyncio
    import pm_coder
    from pydantic_ai import Agent
    from pydantic_ai.models.function import FunctionModel

    workspace = tmp_path / "repo"
    workspace.mkdir()
    skill = tmp_path / "external" / "SKILL.md"
    skill.parent.mkdir()
    body = "---\nname: external-role\ndescription: Test role\n---\nAlways write evidence before committing."
    skill.write_text(body, encoding="utf-8")
    settings = pm_coder.build_settings(cwd=workspace, model="test", context_window=96000, skill=str(skill))
    observed = []

    def respond(messages, info):
        observed.append(info.instructions)
        return ModelResponse(parts=[TextPart(content="done")])

    async def summary(settings, history):
        return "Task remains; all exploratory reads complete."

    monkeypatch.setattr(pm_coder, "summarize", summary)
    monkeypatch.setattr(pm_coder, "active_session", None)

    async def run():
        instructions = pm_coder.build_system_prompt(settings, pm_coder.discover_workspace(settings))
        agent = Agent(FunctionModel(respond), instructions=instructions)
        result = await agent.run("Implement one item")
        history = await pm_coder.compact(settings, result.all_messages(), 0)
        assert history[0] == result.all_messages()[0]
        await agent.run("Continue", message_history=history)
        # A restarted session reconstructs its instructions from the skill path.
        restarted = Agent(FunctionModel(respond), instructions=pm_coder.build_system_prompt(
            settings, pm_coder.discover_workspace(settings)))
        await restarted.run("Continue", message_history=history)

    asyncio.run(run())
    assert len(observed) == 3
    assert all(body in instructions for instructions in observed)
