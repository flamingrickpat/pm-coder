# pm-coder

## minimal coding agent with mcp and skills for local llms like qwen 3.6

Unattended coding agent for local OpenAI-compatible models, with MCP
discovery, project instructions, skills, and read/write/edit/shell tools.
Built to be started once and left running: no request limits, no wall-clock
limits, and a single recovery loop that compacts its own context and
reconnects forever.

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
pm-coder
  cwd:            C:\Users\you\project
  endpoint:       http://127.0.0.1:8080/v1
  model:          qwen
  shell:          powershell (C:\...\powershell.exe)
  mcp servers:    none configured
  skills:         9
  selected skill: (none)
  instructions:   0 file(s)
  temperature:    0.7
  max tokens:     8192
  context window: 96,256
  verbose:        off
  run id:         2026-08-21_10-54-29_C_Users_you_project
  session dir:    C:\Users\you\.pm\pm-coder\2026-08-21_10-54-29_C_Users_you_project
```

Everything pm-coder says about itself goes to **stderr**. Only the turn's
result goes to stdout, so `--mode auto` output stays parseable even while the
shell tool is echoing commands. Both streams are forced to UTF-8, because
Windows otherwise picks the console codepage and a single em-dash in a model
response is enough to make the JSON invalid for whatever is parsing it. If you
call the Python API instead and redirect stderr, do the same in your own
process.

## Tools

Four built-in tools, plus whatever the MCP servers contribute:

| Tool | Signature | Notes |
| --- | --- | --- |
| `read` | `read(path, offset=1, limit=0)` | Line numbers are prepended for reference. `limit=0` reads the whole file. |
| `write` | `write(path, content)` | Creates a file or fully rewrites one. Always UTF-8, no BOM. |
| `edit` | `edit(path, old_string, new_string, replace_all=False)` | Exact string match, must be unique unless `replace_all`. |
| `powershell` / `bash` | `(command, timeout_seconds)` | Everything else: running, searching, verifying. Named after the selected backend. |

Files are addressed by **exact content, never by line number**. Every serious
agent harness converged on this independently -- aider dropped line numbers
from its diff hunks, Anthropic's editor tool uses `old_str`/`new_str`, and
OpenAI's `apply_patch` anchors on surrounding context -- because a wrong
string fails loudly and can be retried, while a wrong line range *succeeds*
and deletes the wrong code. That distinction decides whether an unattended
overnight run degrades or corrupts. Line numbers appear only in `read` output,
to be quoted back verbatim.

`edit` reports the closest matching lines when `old_string` does not match, so
a model that missed by one level of indentation can fix it in one step instead
of re-reading the file. A file's existing line endings are preserved, so
editing a CRLF file on Windows does not rewrite every line.

`write`/`edit` exist because doing this through the shell forces the model to
escape the same content twice -- once for the tool-call JSON, once for the
shell -- and PowerShell 5.1's `Set-Content` silently writes the system ANSI
codepage rather than UTF-8.

## Options

Every option has a `--flag` and most read an environment variable. The CLI
default is listed in `( )`. Options marked *environment* read from the named
variable when the flag is not passed.

| Option | Default | Environment | What it does |
| --- | --- | --- | --- |
| `--mode {interactive,auto}` | `interactive` | &mdash; | Run a persistent chat session (interactive), or one prompt that runs to completion and exits (auto). |
| `--auto` | &mdash; | &mdash; | Shortcut for `--mode auto`. |
| `--prompt` (positional) | &mdash; | &mdash; | Prompt text (auto mode) or a path to a UTF-8 text file. |
| `--prompt-file PATH` | &mdash; | &mdash; | Read the auto-mode prompt from this file. |
| `--run-id RUN_ID` | new session | &mdash; | Resume the session directory `<log-root>/<RUN_ID>`. |
| `--log-root DIR` | `~/.pm/pm-coder` | &mdash; | Session log root. |
| `--cwd DIR` | current dir | &mdash; | Working directory the agent acts inside. MCP/skill/instruction discovery walks up from here. |
| `--base-url URL` | `http://127.0.0.1:8080/v1` | `LOCAL_AGENT_BASE_URL`, `OPENAI_BASE_URL` | OpenAI-compatible endpoint. |
| `--api-key KEY` | `local` | `LOCAL_AGENT_API_KEY`, `OPENAI_API_KEY` | API key for the endpoint. |
| `--model MODEL` | first `/models` entry | `LOCAL_AGENT_MODEL`, `OPENAI_MODEL` | Model id. When omitted, pm-coder waits for the endpoint and takes the first model it advertises. |
| `--mcp-config PATH` | auto-discover | &mdash; | MCP config file. Discovery walks up looking for `.mcp.json`, `mcp.json`, `mcp_config.json`, `.pi/mcp.json`, `.codex/mcp.json`. |
| `--shell {auto,powershell,bash}` | `auto` | `LOCAL_AGENT_SHELL` | Host shell tool. `auto` picks PowerShell on Windows, Bash elsewhere. |
| `--shell-timeout SECS` | `240` | `LOCAL_AGENT_SHELL_TIMEOUT` | Seconds allowed for one host-shell tool call. `0` means no timeout. |
| `--skill NAME_OR_PATH` | none | &mdash; | Load exactly one skill and inject its full `SKILL.md` into the system prompt, replacing the skill index. Accepts the skill's name or a path to its `SKILL.md`. |
| `--context-window N` | `0` (ask the endpoint) | `LOCAL_AGENT_CONTEXT_WINDOW` | Token budget used to size compaction. `0` reads the endpoint's advertised `n_ctx` and subtracts a 2048-token safety margin; if the endpoint does not advertise one, 65536 is assumed. |
| `--temperature T` | `0.7` | `LOCAL_AGENT_TEMPERATURE` | Sampling temperature. |
| `--max-tokens N` | `8192` | `LOCAL_AGENT_MAX_TOKENS` | Generated tokens per model response. `0` lets the server decide. |
| `--enable-thinking` / `--disable-thinking` | enabled | &mdash; | Allow or disable provider-specific long-form reasoning. |
| `-v` / `--verbose` | off | &mdash; | Print the raw model stream to stderr as it arrives: thinking, text, and tool-call arguments exactly as they come in, plus request markers and usage. Useful for diagnosing a hang before anything is persisted. |

There are deliberately no request limits, wall-clock limits, or output caps.
The agent is meant to be started once and left alone.

## Recovery

Every model turn -- interactive or scripted -- goes through one function,
`run_turn`, which has three failure policies and no exit condition:

- **Out of context.** The endpoint rejected the request for length, or a
  response came back truncated with the context above 75% full. The older
  history is summarized into a checkpoint and the turn resumes from the last
  tool result, so the model continues instead of restarting. Each compaction
  that has to happen again immediately halves the verbatim context it
  preserves; one that follows real progress starts over at full detail.
- **Out of response budget.** A response was truncated (`finish_reason ==
  "length"`) while the context still had room -- it simply outgrew
  `--max-tokens`. Summarizing cannot fix that, so pm-coder does not: whatever
  the model produced is kept, and it is told to continue from where it stopped
  and to split large writes. A response that stopped *inside* a tool call is
  dropped instead, since its arguments are incomplete.
- **Anything else.** Connection refused, a restarted server, an HTTP error, a
  model that changed underneath you: pm-coder prints
  `<error>, trying reconnect...`, waits 30 seconds, and retries the same turn
  from the work already captured. Tool calls that already ran are not repeated.

Nothing but Ctrl-C ends the loop. Startup blocks the same way: if the endpoint
is not up yet and the model id or context window still has to be discovered,
pm-coder waits for it rather than starting against a server that is not there.

Compaction has three strategies that escalate in order: summarize the interior
of any single turn larger than 20% of the context window; otherwise keep whole
turns at both ends and summarize the whole turns in between; otherwise collapse
the largest turn down to its original request plus a checkpoint. Images in tool
results are dropped first -- they are the cheapest thing to lose. If the
summarizer itself runs out of context, its input is halved with overlap and
summarized recursively.

## Sessions

Auto mode writes one JSON object to stdout:

```json
{"response":"...","run_id":"2026-08-21_10-54-29_C_source_project","duration_seconds":12.34,"tokens_used":{"input_tokens":3413,"output_tokens":297,"requests":3,"tool_calls":2}}
```

Sessions live in `~/.pm/pm-coder/<run_id>/`:

- `messages.json` -- Pydantic AI's structured model messages, not a summary or
  a second prompt. Replayed verbatim into `message_history`, so resuming does
  not re-prompt anything. Pass `--run-id <run_id>` to continue. It is rewritten
  **mid-turn**, after a tool call and at most every 30 seconds, so a turn that
  runs all night survives a crash: kill the process at hour nine and
  `--run-id` picks up from the last completed tool call.
- `runs.jsonl` -- one line per completed turn.
- `session.json` -- run metadata.
- `<timestamp>_<n>.compact.json` / `.pretty.json` -- the exact body of every
  `/chat/completions` request, captured below Pydantic AI's message and tool
  conversion. One pair per request, so a long session produces a lot of them.

With the same model, tokenizer, system instructions, tools, and chat-template
settings, a resumed session sends the same logical prompt tokens. The llama.cpp
server's private in-memory KV cache cannot be persisted through the
OpenAI-compatible API.

Interactive commands: `/clear`, `/info`, `/paste`, `/quit`.

## Minecraft example (pm-minecraft)

Point pm-coder at a pm-minecraft character workspace (e.g. `C:/Temp/Floppa`).
pm-coder auto-discovers that workspace's `.mcp.json`, connects to the
`minecraft` MCP server, and lets the model drive the body while reading
screenshots it captures:

```powershell
pm-coder --mode auto `
  --cwd C:/Temp/Floppa `
  "Use the minecraft MCP server. Start with minecraft_observe(include_image=true) to actually look at the screenshot of your surroundings. Get a diamond. Use normal survival mechanics only."
```

- `--cwd C:/Temp/Floppa` makes pm-coder discover `C:/Temp/Floppa/.mcp.json`
  automatically (no `--mcp-config` needed).
- `minecraft_observe` with `include_image=true` returns image pixels a
  multimodal model can inspect directly. Those images are the first thing
  compaction discards when context runs low.
- The workspace's `AGENTS.md` and its `drafts/*.ts` teach the model the exact
  tool shapes and safe exploration habits.
- The MCP connection is opened once and held for the whole session. A model
  endpoint that dies and comes back does not disturb it, so the bot is not
  kicked from the world every time llama.cpp restarts.

## Python API

`run_auto` has the same behavior as auto CLI mode and returns the result as a
plain dictionary. Reuse its `run_id` for a continuing conversation:

```python
from pm_coder import run_auto

run_id = None
while True:
    prompt = input("You> ")
    if not prompt:
        break
    result = run_auto(prompt, run_id=run_id)
    run_id = result["run_id"]
    print(result["response"])
```

`run_auto` owns the event loop and returns a dictionary, so callers never touch
asyncio. Use `async_run_auto` from an existing asyncio application; it returns
a `TurnResult`.

Both accept the same keywords as the CLI flags -- `cwd`, `base_url`, `api_key`,
`model`, `mcp_config`, `shell`, `shell_timeout`, `temperature`, `max_tokens`,
`enable_thinking`, `skill`, `verbose`, `context_window` -- plus `run_id` and
`log_root`.

To run many turns against one live agent and one MCP connection, use
`open_session` directly:

```python
import asyncio
from pm_coder import build_settings, open_session, run_turn

async def main():
    settings = build_settings(cwd="C:/Temp/Floppa")
    async with open_session(settings) as (agent, discovery, store):
        while True:
            await run_turn(agent, settings, store, "keep mining until you find diamonds")

asyncio.run(main())
```


## Development

```powershell
python -m pip install -e "."
```

There is no test suite. The agent is verified by running it.

