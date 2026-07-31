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

Auto mode writes one JSON object to stdout:

```json
{"response":"...","run_id":"2026-07-31_19-00-00_C_source_project","duration_seconds":12.34,"tokens_used":{"requests":1,"input_tokens":123,"output_tokens":45,"total_tokens":168}}
```

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
`http://127.0.0.1:8080/v1` wiht `qwen`  model by default. 
Use the command-line options to select a
model, working directory, MCP configuration, shell, and execution limits.
Thinking and response length are unlimited by this client by default. The
server's context window, EOS, and any explicit `--max-tokens` or
`--disable-thinking` option remain the effective boundaries.

Interactive commands include `/clear`, `/info`, `/paste`, and `/quit`.

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
name for callers that want to make the threading boundary clear.

## Development

```powershell
python -m pip install -e ".[dev]"
pytest
```
