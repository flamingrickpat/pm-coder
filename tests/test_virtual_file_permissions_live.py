"""A real agent reads a virtual prompt and uses its assigned user's file tools."""
import json
import os

import httpx

from pm_bash_machine import Access, BashMachine
from pm_coder import run_auto_with_bash_machine


def test_agent_uses_virtual_prompt_and_nondefault_user(tmp_path):
    endpoint = os.environ.get("LOCAL_AGENT_BASE_URL", "http://127.0.0.1:8080/v1")
    model = os.environ.get("LOCAL_AGENT_MODEL", "qwen")
    key = os.environ.get("LOCAL_AGENT_API_KEY", "local")
    with httpx.Client(timeout=10) as client:
        response = client.get(endpoint.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"})
        response.raise_for_status()
        assert model in {item["id"] for item in response.json()["data"]}
    config = tmp_path / "empty-mcp.json"
    config.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    machine = BashMachine()
    machine.write_text("/work/memory.txt", "WORKFLOW_MEMORY_56291", access={"workflow": Access.R, "default": Access.N})
    machine.write_text("/work/request.md", "Use the read tool on memory.txt. Then use the write tool to copy its exact content "
                       "into result.txt. Do not use Bash or subagents. Do not change memory.txt. Reply done.", access=Access.R)
    machine.add_user("workflow", cwd="/work")
    result = run_auto_with_bash_machine("request.md", machine, user="workflow", cwd=tmp_path,
                                      base_url=endpoint, model=model, api_key=key, context_window=96_000,
                                      enable_thinking=False, mcp_config=config, log_root=tmp_path / "logs", run_id="workflow")
    assert result["response"]
    # This checks tool routing, not the model's choice of a final newline.
    assert machine.read_binary_as("workflow", "result.txt").splitlines() == [b"WORKFLOW_MEMORY_56291"]
    assert machine.read_text("/work/memory.txt") == "WORKFLOW_MEMORY_56291"
    assert not machine.is_file("/home/user/result.txt")
    assert not (tmp_path / "result.txt").exists()
    messages = json.loads((tmp_path / "logs" / "workflow" / "messages.json").read_text(encoding="utf-8"))
    tools = {part["tool_name"] for message in messages for part in message["parts"] if part["part_kind"] == "tool-call"}
    assert {"read", "write"}.issubset(tools)
