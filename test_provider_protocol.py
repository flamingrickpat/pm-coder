"""Preserve reasoning blocks from a captured real hosted tool response.

These checks parse and serialize original protocol data without inference.
The separate live file-tool probe establishes real provider execution.
"""
import asyncio
import json
from pathlib import Path

from openai.types.chat import ChatCompletion
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import ModelRequestParameters
from pm_coder import Settings, is_openrouter_endpoint, make_model


def test_captured_reasoning_blocks_survive_saved_history(tmp_path):
    source = Path(__file__).parent/'tests/fixtures/openrouter/native-tool-response.json'
    original = json.loads(source.read_text(encoding='utf-8'))
    settings = Settings(cwd=tmp_path, base_url='https://openrouter.ai/api/v1',
        api_key='unused-for-offline-parsing', model=original['model'], mcp_config=None,
        shell_kind='powershell', shell_executable='powershell', shell_timeout=5,
        disable_thinking=False, skill=None, verbose=False, context_window=1000000,
        enable_write=False, workspace_discovery=False, live_test=True)
    model = make_model(settings, 'captured-protocol')
    response = model._process_response(ChatCompletion.model_validate(original))
    saved = ModelMessagesTypeAdapter.dump_json([response])
    restored = ModelMessagesTypeAdapter.validate_json(saved)
    async def replay():
        try:
            return await model._map_messages(restored, ModelRequestParameters())
        finally:
            await model.client.close()
    messages = asyncio.run(replay())
    actual = next(message for message in messages if message['role'] == 'assistant')
    expected = original['choices'][0]['message']
    assert [tool['id'] for tool in actual['tool_calls']] == [tool['id'] for tool in expected['tool_calls']]
    blocks = actual['reasoning_details']
    assert len(blocks) == len(expected['reasoning_details'])
    for original_block, replayed in zip(expected['reasoning_details'], blocks):
        assert all(key in replayed and replayed[key] == value for key, value in original_block.items())
        assert all(key in {'id', 'signature', 'format', 'index'} and value is None
            for key, value in replayed.items() if key not in original_block)


def test_protocol_selection_uses_host_identity():
    assert is_openrouter_endpoint('https://openrouter.ai/api/v1')
    assert not is_openrouter_endpoint('https://openrouter.ai.example.com/api/v1')
    assert not is_openrouter_endpoint('http://127.0.0.1:8080/openrouter.ai')
