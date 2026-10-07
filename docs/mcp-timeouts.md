# MCP request timeouts

## Summary

The native agent honors `requestTimeoutMs` for each MCP server.
The value uses milliseconds and must be finite and positive.
The transport retains its original environment, headers, and process arguments.
The agent keeps its normal context discovery and automatic compaction.

Parent: [package overview](index.md).
API: [`pm_coder.load_timed_mcp_toolsets`][].

## Input and execution

The existing `.mcp.json` configuration selects HTTP or stdio servers.
The normal Pydantic AI loader expands environment variables and constructs their transports.
`load_timed_mcp_toolsets` retains each expanded transport.
A server with `requestTimeoutMs` receives that request timeout in seconds.
An omitted value retains the normal loader's timeout.
An invalid value raises `ValueError` before a connection starts.

```json
{"mcpServers":{"minecraft_actor":{"url":"http://127.0.0.1:6767/mcp","requestTimeoutMs":660000}}}
```

PM's Minecraft body permits a 600-second navigation call.
The actor uses 660 seconds for its transport, including result and cleanup reads.
The ordinary five-minute timeout can expire before that real call finishes.
A later model retry then encounters the still-active command.
This configuration repair preserves the original external call and its receipt.
It adds no automatic mutation retry or tool-call ceiling.

## Cancellation and evidence

The caller still owns cancellation and external cleanup.
A timeout cannot roll back a tool's effects.
Minecraft's gateway retains the dispatched command and rejects overlapping mutation plans.
Its epoch fence rejects obsolete sessions after replacement.
Its cancellation watcher sends an owned stop and retains the actual outcome.

The A12 application manual records the real-server failure and repaired gate.
The preserved first native run is `.runtime/alpha/a12-native-01/` in `C:/source/pm/pm_next_v2`.
Its navigation dispatch continued after the old loader's request timeout.
Configuration tests also exercise the actual loader without opening a model connection.
