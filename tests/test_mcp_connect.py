"""Tests for runtime MCP connection via the mcp_connect tool."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pydantic_ai.exceptions import RunCancelled

import pm_coder
from pm_coder import (
    MCPRegistry,
    build_agent,
    build_settings,
    discover_workspace,
    make_mcp_connect_tool,
    mcp_continuation_prompt,
    parse_args,
)


class _FakeCtx:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class _FakeToolset:
    """Stand-in for an MCPToolset that needs no real MCP server."""

    server_info = SimpleNamespace(name="demo", version="1.0")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_args = args
        self.init_kwargs = kwargs

    async def list_tools(self) -> list[Any]:
        return [SimpleNamespace(name="greet", description="say hello")]

    async def __aenter__(self) -> "_FakeToolset":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class _RaisingToolset:
    """Toolset whose connection is attempted but fails."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_kwargs = kwargs

    async def list_tools(self) -> list[Any]:
        raise ConnectionError("connection refused by the server")

    async def __aenter__(self) -> "_RaisingToolset":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


def test_registry_tracks_new_connections() -> None:
    registry = MCPRegistry()
    assert registry.new_connections() == []
    registry.add(
        "http://a/mcp",
        _FakeToolset("http://a/mcp"),
        server_name="alpha",
        server_version="2.0",
        tool_names=["one", "two"],
    )
    registry.add(
        "http://b/mcp",
        _FakeToolset("http://b/mcp"),
        server_name="beta",
        server_version="",
        tool_names=[],
    )
    assert registry.urls() == {"http://a/mcp", "http://b/mcp"}
    assert len(registry.toolsets()) == 2
    assert registry.new_connections() == ["http://a/mcp", "http://b/mcp"]
    assert "alpha (2.0)" in registry.describe("http://a/mcp")
    assert "one, two" in registry.describe("http://a/mcp")
    registry.mark_built()
    assert registry.new_connections() == []
    assert registry.urls() == {"http://a/mcp", "http://b/mcp"}


def test_connect_tool_rejects_non_http_url() -> None:
    registry = MCPRegistry()
    connect = make_mcp_connect_tool(registry).function
    ctx = _FakeCtx()
    result = __import__("asyncio").run(connect(ctx, "stdio-server.py"))
    assert result.startswith("error: mcp_connect only supports http:// or https://")
    assert registry.urls() == set()
    assert ctx.cancelled is False


def test_connect_tool_rejects_bad_headers() -> None:
    registry = MCPRegistry()
    connect = make_mcp_connect_tool(registry).function
    ctx = _FakeCtx()
    result = __import__("asyncio").run(
        connect(ctx, "http://example.com/mcp", headers_json="not json")
    )
    assert result.startswith("error: headers_json must be a JSON object")
    assert registry.urls() == set()
    assert ctx.cancelled is False


def test_connect_tool_returns_error_when_connection_fails(
    monkeypatch: Any,
) -> None:
    registry = MCPRegistry()
    monkeypatch.setattr(pm_coder, "MCPToolset", _RaisingToolset)
    connect = make_mcp_connect_tool(registry).function
    ctx = _FakeCtx()
    result = __import__("asyncio").run(connect(ctx, "http://dead/mcp"))
    assert result.startswith("error: could not connect to MCP server at")
    assert "connection refused" in result  # via _mcp_error_text
    assert registry.urls() == set()
    assert ctx.cancelled is False


def test_connect_tool_success_registers_and_cancels(monkeypatch: Any) -> None:
    registry = MCPRegistry()
    monkeypatch.setattr(pm_coder, "MCPToolset", _FakeToolset)
    connect = make_mcp_connect_tool(registry).function
    ctx = _FakeCtx()
    result = __import__("asyncio").run(connect(ctx, "http://demo/mcp"))
    assert registry.urls() == {"http://demo/mcp"}
    assert ctx.cancelled is True
    assert "demo (1.0)" in result
    assert "greet" in result


def test_connect_tool_dedupes_existing_connection(monkeypatch: Any) -> None:
    registry = MCPRegistry()
    registry.add(
        "http://demo/mcp",
        _FakeToolset("http://demo/mcp"),
        server_name="demo",
        server_version="1.0",
        tool_names=["greet"],
    )
    monkeypatch.setattr(pm_coder, "MCPToolset", _FakeToolset)
    connect = make_mcp_connect_tool(registry).function
    ctx = _FakeCtx()
    result = __import__("asyncio").run(connect(ctx, "http://demo/mcp"))
    assert "already connected" in result
    assert ctx.cancelled is False  # no rebuild request for an existing server
    assert registry.urls() == {"http://demo/mcp"}


def test_continuation_prompt_lists_pending_connections() -> None:
    registry = MCPRegistry()
    registry.add(
        "http://demo/mcp",
        _FakeToolset("http://demo/mcp"),
        server_name="demo",
        server_version="1.0",
        tool_names=["greet"],
    )
    prompt = mcp_continuation_prompt(registry)
    assert prompt.startswith("Please continue.")
    assert "demo (1.0)" in prompt


def test_run_cancelled_is_not_a_transient_endpoint_failure() -> None:
    # mcp_connect cancels the run; the host must handle it as a rebuild signal,
    # never misclassify it as an endpoint outage and retry the same prompt.
    assert pm_coder._is_transient_endpoint_failure(RunCancelled("stop")) is False


def test_build_agent_includes_registry_toolset(tmp_path: Any) -> None:
    args = parse_args(["--cwd", str(tmp_path)])
    settings = build_settings(args)
    discovery = discover_workspace(settings)
    registry = MCPRegistry()
    toolset = pm_coder.MCPToolset("http://demo/mcp")
    registry.add(
        "http://demo/mcp",
        toolset,
        server_name="demo",
        server_version="1.0",
        tool_names=["greet"],
    )
    agent = build_agent(settings, discovery, mcp_registry=registry)
    assert toolset in agent.toolsets
