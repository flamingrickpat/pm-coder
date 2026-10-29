"""Real stores, tools, and HTTP request logging without a process-global session."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

import httpx
from pydantic_ai.providers.openai import OpenAIProvider

from pm_coder import (
    LoggingOpenAIChatModel, SessionStore, build_settings, context_limits, make_file_tools, make_model,
)


def test_context_limits_are_explicit_and_support_small_contexts():
    assert context_limits(32_000).read_lines == 128
    assert context_limits(96_000).read_lines == 256
    assert context_limits(1024) == context_limits(32_000)


def test_model_has_no_generation_timeout(tmp_path):
    settings = build_settings(cwd=tmp_path, model="unused", context_window=96_000)
    model = make_model(settings, "test")
    assert model.client.timeout is None
    asyncio.run(model.client.close())


def test_concurrent_file_tools_keep_their_own_limits(tmp_path):
    (tmp_path / "data.txt").write_text("sample content\n" * 10_000, encoding="utf-8")
    readers = []
    for context in (32_000, 96_000):
        settings = build_settings(cwd=tmp_path, model="unused", context_window=context)
        readers.append(next(tool.function for tool in make_file_tools(settings) if tool.name == "read"))
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda read: read("data.txt", line_length=10_000), readers))
    assert "HARD OUTPUT LIMIT" in results[0]
    assert len(results[0]) <= context_limits(32_000).read_output_chars
    assert len(results[1]) <= context_limits(96_000).read_output_chars
    assert len(results[1]) > len(results[0]) * 1.5


def test_request_logs_stay_in_the_supplied_store_and_keep_every_request(tmp_path):
    async def exercise():
        stores = [SessionStore.open(tmp_path, name, log_root=tmp_path / "logs") for name in ("first", "second")]
        models = [LoggingOpenAIChatModel("qwen", session=store,
                  provider=OpenAIProvider(base_url="http://127.0.0.1:8080/v1", api_key="local")) for store in stores]
        try:
            for index in range(2):
                await asyncio.gather(*[model._log_http_request(httpx.Request(
                    "POST", "http://127.0.0.1:8080/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": f"{store.run_id}-{index}"}]}))
                    for store, model in zip(stores, models)])
            for store in stores:
                logs = list(store.path.glob("turn_*.json"))
                assert len(logs) == 2
                messages = {json.loads(path.read_text())["messages"][0]["content"] for path in logs}
                assert messages == {f"{store.run_id}-0", f"{store.run_id}-1"}
        finally:
            for model in models:
                await model.client.close()
    asyncio.run(exercise())


def test_concurrent_automatic_run_ids_have_separate_directories(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as workers:
        stores = list(workers.map(lambda _: SessionStore.open(tmp_path, log_root=tmp_path / "logs"), range(32)))
    assert len({store.path for store in stores}) == 32
    assert all(store.metadata_path.is_file() for store in stores)
