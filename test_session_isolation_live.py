"""Required live session isolation. No mocked model and no generation deadline."""
import asyncio
import json
import os

import httpx

from pm_coder import build_settings, compact, open_session, run_turn


def test_overlapping_live_sessions_keep_prompts_logs_and_turn_state(tmp_path):
    endpoint = os.environ.get("LOCAL_AGENT_BASE_URL", "http://127.0.0.1:8080/v1")
    model = os.environ.get("LOCAL_AGENT_MODEL", "qwen")
    key = os.environ.get("LOCAL_AGENT_API_KEY", "local")
    # An unavailable endpoint fails this test before the production reconnect loop.
    with httpx.Client(timeout=10) as client:
        response = client.get(endpoint.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"})
        response.raise_for_status()
        assert model in {item["id"] for item in response.json()["data"]}
    config = tmp_path / "empty-mcp.json"
    config.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    settings = build_settings(cwd=tmp_path, base_url=endpoint, model=model, api_key=key,
                              context_window=96_000, enable_thinking=False, enable_write=False, mcp_config=config)
    markers = ["ALPHA_SESSION_ONLY_5821", "BETA_SESSION_ONLY_9246"]

    async def exercise():
        async with open_session(settings, run_id="alpha", log_root=tmp_path / "logs") as (first, _, first_store):
            async with open_session(settings, run_id="beta", log_root=tmp_path / "logs") as (second, _, second_store):
                turns = await asyncio.gather(*[
                    run_turn(agent, settings, store, f"Return exactly {marker}. Do not call any tools.")
                    for agent, store, marker in zip((first, second), (first_store, second_store), markers)
                ])
                checkpoints = await asyncio.gather(*[
                    compact(settings, store.load_messages(), 0, store) for store in (first_store, second_store)
                ])
                assert all(checkpoints)
        for index, (store, turn) in enumerate(zip((first_store, second_store), turns)):
            assert markers[index] in turn.response
            assert store.turn_id
            assert store.auto_compact_cnt == 0
            assert turn.run_id == store.run_id
            requests = list(store.path.glob("turn_*.json"))
            assert requests
            text = "\n".join(path.read_text(encoding="utf-8") for path in requests)
            assert markers[index] in text
            assert markers[1 - index] not in text
            assert all(path.name.startswith(f"turn_{store.turn_id}_ac_0_") for path in requests)
            assert markers[index] in store.messages_path.read_text(encoding="utf-8")
            assert markers[1 - index] not in store.messages_path.read_text(encoding="utf-8")
            assert any("_compact_" in path.name for path in requests)
        assert first_store.turn_id != second_store.turn_id

    asyncio.run(exercise())
