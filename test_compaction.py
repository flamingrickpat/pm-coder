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


def test_repeated_recovery_can_drop_even_the_last_exchange():
    history = [ModelRequest(parts=[UserPromptPart(content="task" * 10_000)])]
    history += exchange("recent", "x" * 2_000)
    assert compaction_tail(history, 0)
    assert compaction_tail(history, 16) == []
