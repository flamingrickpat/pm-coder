# pm-coder

## minimal coding agent with mcp and skills for local llms like qwen 3.6

Bounded coding agent for local OpenAI-compatible models, with MCP discovery,
project instructions, skills, and a host-shell tool.

## Install

Install the latest repository version in another project:

```powershell
python -m pip install --upgrade "pm-coder @ git+https://github.com/flamingrickpat/pm-coder"
```

The package installs the `pm-coder` command and the `pm_coder` Python module.

## Run

```powershell
pm-coder --help
pm-coder                         # interactive mode
pm-coder --auto "Inspect the current project"
pm-coder --mode auto prompt.txt
```

When you start pm-coder it prints its effective configuration so you can
verify it is set up the way you expect:

```text
private-machine-coder
  cwd:                        C:\Users\you\project
  endpoint:                   http://127.0.0.1:8080/v1
  model:                      qwen
  shell:                      powershell (C:\...\powershell.exe)
  skills:                     9
  instructions:               0 file(s)
  MCP servers:                none configured
  temperature:                0.1
  max tokens:                 8192
  selected skill:             (none)
  auto-compact:               off
```

## Options

Every option has a `--flag` and most read an environment variable. The CLI
default is listed in `( )`. Options marked *environment* read from the named
variable when the flag is not passed.

| Option | Default | Environment | What it does |
| --- | --- | --- | --- |
| `--mode {interactive,auto}` | `interactive` | &mdash; | Run a persistent chat session (interactive) or one JSON-producing prompt (auto). |
| `--auto` | &mdash; | &mdash; | Shortcut for `--mode auto`. |
| `--prompt` (positional) | &mdash; | &mdash; | Prompt text (auto mode) or a path to a UTF-8 text file. |
| `--run-id RUN_ID` | none | &mdash; | Resume the session directory `~/.pm/pm-coder/<RUN_ID>`. |
| `--log-root DIR` | `~/.pm/pm-coder` | &mdash; | Override the session log root (mainly for tests). |
| `--cwd DIR` | current dir | &mdash; | Working directory the agent acts inside. MCP/skill/instruction discovery walks up from here. |
| `--base-url URL` | `http://127.0.0.1:8080/v1` | `LOCAL_AGENT_BASE_URL`, `OPENAI_BASE_URL` | OpenAI-compatible endpoint. |
| `--api-key KEY` | `local` | `LOCAL_AGENT_API_KEY`, `OPENAI_API_KEY` | API key for the endpoint. |
| `--model MODEL` | detected from `/models` | `LOCAL_AGENT_MODEL`, `OPENAI_MODEL` | Model id. Falls back to the first `/models` result, else `local`. |
| `--mcp-config PATH` | auto-discover | &mdash; | MCP config file. Default discovery walks up looking for `.mcp.json`, `mcp.json`, `mcp_config.json`, `.pi/mcp.json`, `.codex/mcp.json`. |
| `--shell {auto,powershell,bash}` | `auto` | `LOCAL_AGENT_SHELL` | Host shell tool. `auto` picks PowerShell on Windows, Bash elsewhere. |
| `--shell-timeout SECS` (`--powershell-timeout`) | `180` | `LOCAL_AGENT_SHELL_TIMEOUT` | Seconds allowed for one host-shell tool call. |
| `--max-tool-output N` | unlimited | `LOCAL_AGENT_MAX_TOOL_OUTPUT` | Cap on shell-tool output characters. |
| `--max-skill-index N` | unlimited | `LOCAL_AGENT_MAX_SKILL_INDEX` | Cap on the skill-index characters injected into context. |
| `--max-project-instructions N` | unlimited | `LOCAL_AGENT_MAX_PROJECT_INSTRUCTIONS` | Cap on instructions (AGENTS.md/CLAUDE.md) characters. |
| `--skill NAME` | none | &mdash; | Load exactly one skill by name and inject its full `SKILL.md` into the system context before the user request. The generic skill index is replaced by that skill's body. |
| `--auto-compact` | off | &mdash; | Compress long sessions. When estimated context exceeds `context-window - compact-reserve-tokens`, the older turns are summarized and only the most recent ~`compact-keep-recent-tokens` are kept verbatim. |
| `--context-window N` | `65536` | `LOCAL_AGENT_CONTEXT_WINDOW` | Estimated model context window in tokens. Used only by `--auto-compact`. |
| `--compact-reserve-tokens N` | `16384` | `LOCAL_AGENT_COMPACT_RESERVE` | Tokens reserved for the next response when deciding to compact (matches pi's default). |
| `--compact-keep-recent-tokens N` | `20000` | `LOCAL_AGENT_COMPACT_KEEP_RECENT` | Approximate tokens of the most recent conversation kept verbatim after compaction (matches pi's default). |
| `--temperature T` | `0.1` | `LOCAL_AGENT_TEMPERATURE` | Sampling temperature. |
| `--max-tokens N` | `8192` | `LOCAL_AGENT_MAX_TOKENS` | Generated tokens per model response. Use `-1` to remove the client response limit. |
| `--enable-thinking` / `--disable-thinking` | enabled | &mdash; | Allow or disable provider-specific long-form reasoning. |
| `--prompt-file PATH` | &mdash; | &mdash; | Prompt source file. With `--artifact-dir` enables legacy one-shot/heartbeat mode. |
| `--artifact-dir DIR` | &mdash; | &mdash; | Legacy one-shot output directory (with `--prompt-file`, `--request-limit`, `--wall-clock-limit`). |
| `--request-limit N` | `0` (unlimited) | &mdash; | Maximum model requests for one turn, counting main-agent **and** compaction-summary calls. `0` = unlimited. |
| `--wall-clock-limit SECS` | `0` (unlimited) | &mdash; | Maximum wall-clock time for one turn. `0` = unlimited. |
| `-v` / `--verbose` | off | &mdash; | Debugging mode: print the raw model generation to stdout as it happens (thinking, text, and tool-call arguments exactly as they arrive), plus request/response markers and usage, and the auto-compaction trigger and summary. Useful for diagnosing hangs, retries, and truncated responses before anything is persisted. |

### `--auto-compact` details

Auto-compact mirrors the upstream pi coding agent's compaction settings and
logic: default `compact-reserve-tokens` of `16384` and
`compact-keep-recent-tokens` of `20000`. When the estimated in-context tokens
exceed `context_window - compact_reserve_tokens`, pm-coder walks backward from
the newest message accumulating estimated sizes, picks a cut point at the
next fresh user-turn boundary (never mid-turn, never at a tool result), and
already-compresses everything older into one structured checkpoint summary
using a short, tool-less LLM pass. The summary is merged into the first kept
user turn so the message stream stays alternating. A failed summarization
never breaks the session: it logs a warning and leaves the history unchanged.

Compaction is off by default; pass `--auto-compact` to enable it. Because a
model's true context window varies, set `--context-window` to match your
model (for example `--context-window 131072` on an 128k model).

Auto mode writes one JSON object to stdout:

```json
{"response":"...","run_id":"2026-07-31_19-00-00_C_source_project","duration_seconds":12.34,"tokens_used":{"requests":1,"input_tokens":123,"output_tokens":45,"total_tokens":168}}
```

With `--verbose`, the raw model stream is also printed to stdout before this
line (one marker per model request, then the generated content as it arrives).
Pipe stdout elsewhere if the extra output breaks your consumer.

Sessions are stored in `~/.pm/pm-coder/<run_id>/`. The `messages.json` file
contains Pydantic AI's structured model messages, not a summary or a second
prompt. Pass `--run-id <run_id>` to continue a session. The same model,
tokenizer, system instructions, tools, and chat-template settings reproduce
the same logical prompt tokens. The llama.cpp server's private in-memory
KV-cache cannot be persisted through the OpenAI-compatible API.

Each session also contains `messages.yaml`, a human-readable rendering with
120-character wrapping and Markdown content preserved in YAML block strings.
Use `messages.json` for replay; use `messages.yaml` for inspection.

The agent uses the local OpenAI-compatible endpoint at
`http://127.0.0.1:8080/v1` with the `qwen` model by default. Use command-line
options to select a model, working directory, MCP configuration, shell, and
execution limits.

The default response limit is 8,192 generated tokens. Use `--max-tokens` to
change it. Use `--max-tokens -1` to remove the client response limit.

Auto and interactive modes recover when a response reaches the client limit.
The agent removes the incomplete response and asks the model to use shorter
tool calls. It uses the completed message and repository state from before the
failure. It also recovers from a known malformed tool-call HTTP 500 response.

Connection failures and restart-class HTTP responses use exponential retry.
The failed endpoint request does not use the agent request budget. The same
wall-clock limit still applies. A real model-context error ends the turn so the
workflow kernel can start a fresh attempt.

Interactive commands include `/clear`, `/info`, `/paste`, and `/quit`.

## Minecraft example (pm-minecraft)

Point pm-coder at a pm-minecraft character workspace (e.g. `C:/Temp/Floppa`).
pm-coder auto-discovers that workspace's `.mcp.json`, connects to the
`minecraft` MCP server, and lets the model drive the body while reading
screenshots it captures. Run it in single-shot auto mode to have the agent
observe its surroundings and report what it sees:

```powershell
pm-coder --mode auto `
  --cwd C:/Temp/Floppa `
  --request-limit 30 `
  --wall-clock-limit 480 `
  "Use the minecraft MCP server. Start with minecraft_observe(include_image=true) to actually look at the screenshot of your surroundings. Scan for a village or any villager. If none is visible from here, say so clearly and describe exactly what you can see, then stop. Use normal survival mechanics only."
```

- `--cwd C:/Temp/Floppa` makes pm-coder discover `C:/Temp/Floppa/.mcp.json`
  automatically (no `--mcp-config` needed).
- The `minecraft` server's `requestTimeoutMs` (200000 in the character
default) is the ceiling for one long-running body/skill call, so keep the
  prompt's own limits generous; `minecraft_observe` with `include_image=true`
  returns image pixels that a multimodal model can inspect directly.
- The workspace's `AGENTS.md` and its `drafts/find_village.ts` teach the model
  the exact tool shapes and safe exploration habits.

## Python API

`run_auto` has the same behavior as auto CLI mode and returns the JSON-compatible
result dictionary. Reuse its `run_id` for a continuing conversation:

```python
from pm_coder import run_auto_sync

run_id = None
while True:
    prompt = input("You> ")
    if not prompt:
        break
    result = run_auto_sync(prompt, run_id=run_id)
    run_id = result["run_id"]
    print(result["response"])
```

Use `async_run_auto` from an existing asyncio application.

`run_auto` is also synchronous; `run_auto_sync` is provided as an explicit
name for callers that want to make the threading boundary clear. Passing
`auto_compact=True`, `skill="..."`, `context_window=...`,
`compact_reserve_tokens=...`, `compact_keep_recent_tokens=...`, and
`verbose=True` to `run_auto` / `run_auto_sync` enables the same behaviors as
the matching CLI flags (`verbose=True` prints the raw model stream to
`sys.stdout` as it happens).

Asyncio is fully opt-in for callers. `run_auto` and `run_auto_sync` run the
whole session (including any auto-compaction) inside a loop they own and
return a plain dictionary; you never touch an event loop. Only
`async_run_auto` exposes the coroutine-based API for embedding in an existing
asyncio application.

## Development

```powershell
python -m pip install -e ".[dev]"
pytest
```
