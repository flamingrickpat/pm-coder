"""Embedded runs must not discover unrelated host instructions or services."""
import socket

import pytest

from pm_coder import build_settings, discover_workspace, make_model, run_auto


def test_explicit_workspace_ignores_ambient_files(tmp_path):
    (tmp_path / "AGENTS.md").write_text("UNRELATED HOST INSTRUCTIONS", encoding="utf-8")
    (tmp_path / ".mcp.json").write_text('{"mcpServers":{"unrelated":{}}}', encoding="utf-8")
    settings = build_settings(cwd=tmp_path, model="explicit", context_window=96000,
                              api_key="", workspace_discovery=False, live_test=True)
    discovery = discover_workspace(settings)
    assert settings.api_key == ""
    assert settings.mcp_config is None
    assert discovery.skills == []
    assert discovery.instruction_files == []
    assert discovery.mcp_server_names == []
    assert make_model(settings, "agent").client.api_key == "local"


def test_explicit_workspace_keeps_explicit_mcp(tmp_path):
    config = tmp_path / "services.json"
    config.write_text('{"mcpServers":{"supplied":{}}}', encoding="utf-8")
    settings = build_settings(cwd=tmp_path, model="explicit", context_window=96000,
                              mcp_config=config, workspace_discovery=False)
    assert discover_workspace(settings).mcp_server_names == ["supplied"]


def test_live_mode_reports_real_connection_failure(tmp_path):
    # Keep the port reserved without listening. No model response is simulated.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        with pytest.raises(Exception, match="[Cc]onnection"):
            run_auto("hello", cwd=tmp_path, model="missing", context_window=96000,
                     base_url=f"http://127.0.0.1:{listener.getsockname()[1]}/v1",
                     api_key="local", workspace_discovery=False, live_test=True,
                     log_root=tmp_path / "logs", run_id="failure")
