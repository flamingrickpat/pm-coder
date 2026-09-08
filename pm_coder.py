r"""Local coding agent built to run unattended for days.

One agent, one host shell, MCP tools, skills, project instructions. Every
model turn -- interactive or scripted -- goes through :func:`run_turn`, which
has three failure policies and no exit condition:

* the context is full
  -> summarize older history into a checkpoint and resume where we left off;
* a response outgrew --max-tokens with context to spare
  -> keep what it wrote and tell it to continue;
* the endpoint could not use what the model produced (a tool call whose JSON
  never parsed, or tool arguments it failed to repair until pydantic-ai's
  retries ran out)
  -> retry, and once that has failed twice running, retry as a
  fresh user turn so the request is not byte-identical;
* anything else
  -> print it, wait, and retry the same turn.

Nothing else stops the loop. There are no request limits, no wall-clock
limits, and no output caps. Values that must exist are used directly so a
logic error crashes loudly instead of being papered over.

Two things ride along inside that loop:

* loop detection -- every tool call is normalized and counted; when the
  last LOOP_WINDOW calls repeat at most LOOP_MAX_DISTINCT operations, a
  fake user turn tells the model to stop, and each compaction hands the
  model a stats view of every call and its amount;
* sub-agents -- the `subagent` tool runs 1-5 fresh pm-coder sessions to
  completion concurrently and returns one report:
  how each finished, a summary of its chat, and its final answer.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import io
import itertools
import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, List
from xml.sax.saxutils import escape, quoteattr

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, Tool, UsageLimits, capture_run_messages
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.mcp import load_mcp_toolsets
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
    UserPromptPart, is_multi_modal_content, ToolReturnPart, ModelMessage, UploadedFile, ImageUrl,
)
from pydantic_ai.models import OpenAIChatCompatibleProvider, StreamedResponse
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIModelName,
)
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles import ModelProfileSpec
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers import Provider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import CombinedToolset, FunctionToolset, WrapperToolset
from pydantic_core import to_jsonable_python
from ruamel.yaml import YAML

APP_NAME = "pm-coder"
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_LOG_ROOT = Path("~/.pm/pm-coder").expanduser()
DEFAULT_SHELL_TIMEOUT = 240
SHELL_TERMINATE_GRACE_SECONDS = 2.0

# Sub-agents: the `subagent` tool accepts this inclusive prompt-count range.
SUBAGENT_MIN_PROMPTS = 1
SUBAGENT_MAX_PROMPTS = 5

# Loop detection: when the last LOOP_WINDOW tool calls hold at most
# LOOP_MAX_DISTINCT distinct calls, the agent is stuck. Two distinct calls
# catch the ping-pong loop (read A, grep B, read A, grep B) a small model
# falls into, which a same-call-only check would miss.
LOOP_WINDOW = 6
LOOP_MAX_DISTINCT = 2

# Seconds to wait before retrying after any failure that is not a context
# problem. One fixed delay: a week-long run has no deadline to race.
RETRY_DELAY_SECONDS = 30.0

# 0 means use the actual serving context advertised by the selected model.
DEFAULT_CONTEXT_WINDOW = 0

@dataclass(frozen=True)
class ContextLimits:
    """Conservative text budgets, in characters unless explicitly named lines.

    These are tuning defaults, not tokenizer estimates or model benchmarks.
    Write size is guidance; exceeding it never discards a generated edit.
    """

    read_lines: int
    read_columns: int
    read_output_chars: int
    shell_lines: int
    shell_chars: int
    write_chunk_chars: int
    compact_tail_chars: int
    compact_summary_chars: int
    summary_overlap_chars: int

    @property
    def read_body_chars(self) -> int:
        return self.read_output_chars - 2_000


# Decimal minimum context sizes: 96_000 and 98_304 both select the 96k row.
# Large contexts grow sublinearly to avoid encouraging whole-repository reads.
#                         read lines/cols/cap, shell lines/cap, write, tail, summary, overlap
CONTEXT_LIMITS = {
    32_000: ContextLimits(128, 256, 12_000, 40, 3_000, 6_000, 16_000, 6_000, 2_000),
    64_000: ContextLimits(192, 512, 20_000, 60, 5_000, 10_000, 32_000, 10_000, 4_000),
    96_000: ContextLimits(256, 512, 28_000, 100, 8_000, 14_000, 48_000, 16_000, 6_000),
    128_000: ContextLimits(384, 768, 36_000, 120, 10_000, 18_000, 64_000, 20_000, 6_000),
    180_000: ContextLimits(512, 1024, 48_000, 160, 12_000, 24_000, 80_000, 24_000, 8_000),
    230_000: ContextLimits(640, 1024, 56_000, 200, 16_000, 28_000, 96_000, 28_000, 8_000),
    512_000: ContextLimits(768, 1536, 80_000, 250, 24_000, 40_000, 128_000, 32_000, 10_000),
    1_000_000: ContextLimits(1024, 2048, 112_000, 300, 32_000, 56_000, 192_000, 48_000, 12_000),
}


def context_limits() -> ContextLimits:
    size = active_session.context_window
    tier = max(minimum for minimum in CONTEXT_LIMITS if size >= minimum)
    return CONTEXT_LIMITS[tier]


# One agent.run can last all night, so the history is snapshotted mid-turn
# after a tool call, at most this often.
SNAPSHOT_SECONDS = 30.0

# Images are pruned from the stored history with a high/low watermark. While
# the history holds at most IMAGE_HIGH_WATER images nothing is touched, so
# consecutive requests differ only by appended messages and prefix caches
# keep hitting; past the watermark, all but the newest IMAGE_LOW_WATER images
# are swapped for placeholders. Swapping content -- never deleting messages --
# keeps every tool-call/tool-return pair intact, so the pruned history can
# never end in unprocessed tool calls.
IMAGE_HIGH_WATER = 32
IMAGE_LOW_WATER = 8
OMITTED = "[older image omitted]"
VISION_REMOVED = "[image removed: this endpoint does not support image input]"

MCP_CONFIG_CANDIDATES = (
    ".mcp.json",
    "mcp.json",
    "mcp_config.json",
    ".pi/mcp.json",
    ".codex/mcp.json",
)
FRONTMATTER_RE = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", re.DOTALL)

# Unlimited must be spelled out: UsageLimits() alone defaults to 50 requests.
NO_LIMITS = UsageLimits(request_limit=None)

# The library surface. Everything here can be imported and used without the
# CLI; anything not listed is an internal detail and may change.
__all__ = [
    "DiscoveryResult",
    "SessionStore",
    "Settings",
    "Skill",
    "TurnResult",
    "async_run_auto",
    "async_run_auto_with_bash_machine",
    "build_agent",
    "build_settings",
    "build_summary_agent",
    "build_system_prompt",
    "compact",
    "discover_workspace",
    "find_mcp_config",
    "find_skill",
    "load_skills",
    "loop_alert_injector",
    "make_bash_machine_tool",
    "make_file_tools",
    "make_shell_tool",
    "make_subagent_tool",
    "open_bash_machine_session",
    "open_session",
    "probe_endpoint",
    "prompt_text",
    "run_auto",
    "run_auto_with_bash_machine",
    "run_turn",
    "select_shell",
    "shell_backend",
    "summarize",
    "wait_for_endpoint",
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def note(message: str) -> None:
    """Narrate to stderr. stdout carries only the turn's result."""
    print(f"{APP_NAME}: {message}", file=sys.stderr, flush=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace one file atomically after flushing its temporary peer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(4):
            try:
                os.replace(temp_path, path)
                break
            except PermissionError:
                # Windows virus scanners hold the handle for a few ms.
                if attempt == 3:
                    raise
                time.sleep(0.01 * (2**attempt))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_yaml(raw: str) -> Any:
    yaml = YAML(typ="safe", pure=True)
    yaml.allow_duplicate_keys = False
    return yaml.load(raw)


# ---------------------------------------------------------------------------
# Session storage
# ---------------------------------------------------------------------------

def _is_image(x) -> bool:
    return (
        isinstance(x, ImageUrl)
        or isinstance(x, (BinaryContent, UploadedFile))
        and x.media_type.startswith("image/")
    )


def count_images(messages: list[ModelMessage]) -> int:
    """Images anywhere in request content, including nested tool returns."""

    def walk(x) -> int:
        if _is_image(x):
            return 1
        if isinstance(x, Mapping):
            return sum(walk(v) for v in x.values())
        if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
            return sum(walk(v) for v in x)
        return 0

    total = 0
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if isinstance(part, UserPromptPart) or (
                isinstance(part, ToolReturnPart) and part.tool_kind is None
            ):
                total += walk(part.content)
    return total


def _omit_older_images(
    messages: list[ModelMessage], keep: int, placeholder: str = OMITTED
) -> list[ModelMessage]:
    """Swap every image but the newest ``keep`` for a placeholder."""

    kept = 0

    # Walk content newest -> oldest.
    def prune(x):
        nonlocal kept

        if _is_image(x):
            if kept < keep:
                kept += 1
                return x
            return placeholder

        # Tool returns can contain arbitrarily nested multimodal data.
        if isinstance(x, Mapping):
            rev = [(k, prune(v)) for k, v in reversed(x.items())]
            return dict(reversed(rev))

        if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
            return list(reversed([prune(v) for v in reversed(x)]))

        return x

    result = []

    # Messages also need to be processed newest -> oldest.
    for msg in reversed(messages):
        if not isinstance(msg, ModelRequest):
            result.append(msg)
            continue

        parts = list(msg.parts)
        changed = False

        for i in range(len(parts) - 1, -1, -1):
            part = parts[i]

            if isinstance(part, UserPromptPart):
                new_content = prune(part.content)

            elif isinstance(part, ToolReturnPart) and part.tool_kind is None:
                new_content = prune(part.content)

            else:
                continue

            parts[i] = replace(part, content=new_content)
            changed = True

        result.append(replace(msg, parts=parts) if changed else msg)

    return list(reversed(result))


def keep_recent_images(messages: list[ModelMessage]) -> list[ModelMessage]:
    """High/low watermark image pruner for the history Pydantic AI stores.

    Runs as a ``ProcessHistory`` capability, and Pydantic AI writes the
    processed list back into the run's message history (the same list our
    mid-turn snapshots persist), so this edits the stored history, not just
    one outgoing request -- which is what makes the watermark stick.

    While the history holds at most ``IMAGE_HIGH_WATER`` images the list is
    returned untouched, so consecutive requests differ only by appended
    messages and prefix caches keep hitting. Past the watermark, all but the
    newest ``IMAGE_LOW_WATER`` images become placeholders -- a content swap,
    never a message removal, so tool-call/tool-return pairing survives and
    the history can never end in unprocessed tool calls.
    """
    total = count_images(messages)
    if total <= IMAGE_HIGH_WATER:
        return messages
    note(f"{total} images in history; keeping the newest {IMAGE_LOW_WATER}")
    return _omit_older_images(messages, IMAGE_LOW_WATER)


def loop_alert_injector(session: SessionStore):
    """Build the ProcessHistory that turns a pending alert into a user turn.

    The alert is appended after the newest message. At this point every tool
    call is answered, so the request stays valid. It is a user turn, not a
    tool retry, because a retried request resamples the same distribution.
    """

    def inject(messages: list[Any]) -> list[Any]:
        alert = session.pending_alert
        if alert is None:
            return messages
        session.pending_alert = None
        note("injecting user turn: " + alert.splitlines()[0])
        return [*messages, ModelRequest(parts=[UserPromptPart(content=alert)])]

    return inject


class SessionStore:
    """One conversation on disk as Pydantic AI model messages.

    ``messages.json`` is replayed verbatim into ``message_history``, so
    resuming a session does not re-summarize or re-prompt anything.
    """

    context_window: int
    schema = "pm-coder-session.v1"

    def __init__(self, path: Path, cwd: Path) -> None:
        self.path = path
        self.cwd = cwd.resolve()
        self.turn_id = ""
        self.auto_compact_cnt = 0
        self.metadata_path = path / "session.json"
        self.messages_path = path / "messages.json"
        self.runs_path = path / "runs.jsonl"
        self.path.mkdir(parents=True, exist_ok=True)
        # Set by run_turn to the list Pydantic AI appends to as a turn runs,
        # so a mid-turn snapshot writes the whole conversation and not just
        # the fragment produced so far.
        self.live_history: list[Any] | None = None
        self.active_stream_path = path / "active-stream.jsonl"
        self._stream_savepoint_counter = itertools.count(1)
        self.last_snapshot = 0.0
        # Normalized tool-call history since the last compaction: loop
        # detection reads the tail, the compaction stats view reads it all.
        self.tool_calls: list[str] = []
        # Text the next model request gets as a fake user turn: a loop alert
        # or the compaction stats view. Cleared once injected.
        self.pending_alert: str | None = None

    def record_tool_call(self, name: str, tool_args: dict[str, Any]) -> None:
        """Count one normalized tool call, and flag a loop when it repeats."""
        key = (name + " " + json.dumps(tool_args, sort_keys=True, default=str)).casefold()
        self.tool_calls.append(key)
        window = self.tool_calls[-LOOP_WINDOW:]
        if len(window) == LOOP_WINDOW and len(set(window)) <= LOOP_MAX_DISTINCT:
            repeated = "".join(f"- {call[:200]}\n" for call in sorted(set(window)))
            self.pending_alert = (
                f"[loop alert] The last {LOOP_WINDOW} tool calls repeated the "
                f"same {len(set(window))} operations:\n{repeated}"
                "Stop this loop. State a new hypothesis, then use a different "
                "tool or different arguments. Do not repeat these calls."
            )

    def tool_stats_report(self) -> str:
        """The compaction stats view: each call and the amount of times."""
        if not self.tool_calls:
            return "(no tool calls since the last checkpoint)"
        counts = Counter(self.tool_calls)
        lines = [f"{amount}x {call[:200]}" for call, amount in counts.most_common()]
        return "tool calls since the last checkpoint (amount x call):\n" + "\n".join(lines)

    @property
    def run_id(self) -> str:
        return self.path.name

    @classmethod
    def open(
        cls,
        cwd: Path,
        run_id: str | None = None,
        *,
        log_root: Path = DEFAULT_LOG_ROOT,
    ) -> SessionStore:
        root = log_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
            cwd_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(cwd.resolve())).strip("_")
            base_name = f"{stamp}_{cwd_id or 'workspace'}"
            path = root / base_name
            suffix = 2
            while path.exists():
                path = root / f"{base_name}-{suffix}"
                suffix += 1
        else:
            if Path(run_id).name != run_id or run_id in {".", ".."}:
                raise ValueError("run_id must be a single safe directory name")
            path = root / run_id
        store = cls(path, cwd)
        if not store.metadata_path.exists():
            atomic_write_bytes(
                store.metadata_path,
                json.dumps(
                    {
                        "schema": cls.schema,
                        "run_id": store.run_id,
                        "created_at": utc_now(),
                        "cwd": str(cwd.resolve()),
                    },
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8"),
            )
        return store

    def load_messages(self) -> list[Any]:
        if not self.messages_path.exists():
            return []
        return ModelMessagesTypeAdapter.validate_json(self.messages_path.read_bytes())

    def save_messages(self, messages: list[Any]) -> None:
        atomic_write_bytes(
            self.messages_path,
            json.dumps(
                to_jsonable_python(messages),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"),
        )

    def snapshot(self) -> None:
        """Persist the in-progress turn, at most every SNAPSHOT_SECONDS."""
        if self.live_history is None:
            return
        now = time.monotonic()
        if now - self.last_snapshot < SNAPSHOT_SECONDS:
            return
        self.last_snapshot = now
        self.save_messages(self.live_history)
        note(f"snapshot: {len(self.live_history)} messages persisted mid-turn")

    def begin_stream_capture(self) -> Any:
        """Reset the main agent's rolling response-stream spool."""
        handle = self.active_stream_path.open("w", encoding="utf-8", newline="\n")
        handle.write(json.dumps({"started_at": utc_now()}) + "\n")
        handle.flush()
        return handle

    def save_stream_savepoint(self) -> Path | None:
        """Copy the last streamed response before compaction discards its context."""
        if not self.active_stream_path.exists():
            return None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target = self.path / (
            f"precompact_{stamp}_{next(self._stream_savepoint_counter):06d}.stream.jsonl"
        )
        shutil.copyfile(self.active_stream_path, target)
        atomic_write_bytes(self.active_stream_path, b"")
        return target

    def clear(self) -> None:
        self.save_messages([])

    def append_run(self, value: dict[str, Any]) -> None:
        with self.runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


# The HTTP logger below sits under Pydantic AI's message layer and has no
# route to the active session, so the store is published here at open time.
active_session: SessionStore | None = None

@dataclass(init=False)
class LoggingOpenAIChatModel(OpenAIChatModel):
    """OpenAIChatModel that dumps every /chat/completions body to the session.

    Logging happens below Pydantic AI's message and tool conversion, so the
    files are exactly what the endpoint received: the raw bytes plus an
    indented copy. Nothing here may break inference.
    """

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: OpenAIChatCompatibleProvider | Provider[AsyncOpenAI],
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        super().__init__(
            model_name, provider=provider, profile=profile, settings=settings
        )
        self._log_counter = itertools.count(1)
        # The only private API involved: AsyncOpenAI's underlying HTTP client.
        hooks = self.client._client.event_hooks
        hooks.setdefault("request", [])
        hooks["request"].append(self._log_http_request)

    async def _log_http_request(self, request: Any) -> None:
        if request.method != "POST":
            return
        if not request.url.path.rstrip("/").endswith("/chat/completions"):
            return
        if active_session is None:
            return

        try:
            raw = bytes(request.content)
            payload = json.loads(raw)
            dump = json.dumps(payload, ensure_ascii=False, indent=2)

            sequence = next(self._log_counter)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            stem = active_session.path / f"turn_{active_session.turn_id}_ac_{active_session.auto_compact_cnt}.json"
            if stem.exists():
                try:
                    os.remove(stem)
                except:
                    stem = stem / f"{timestamp}.json"
            stem.write_text(dump, encoding="utf-8")
        except Exception as exc:
            note(f"prompt logger failed: {exc!r}")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    cwd: Path
    base_url: str
    api_key: str
    model: str
    mcp_config: Path | None
    shell_kind: Literal["powershell", "bash"]
    shell_executable: str
    shell_timeout: int = Field(gt=0)
    disable_thinking: bool
    skill: str | None
    verbose: bool
    context_window: int = Field(gt=0)


def probe_endpoint(
    base_url: str, api_key: str, *, timeout: float = 10.0, model: str | None = None
) -> dict[str, Any] | None:
    """Return the selected model (or first if unspecified), or None if unreachable.

    llama.cpp reports ``meta.n_ctx`` (what a slot can actually fit) alongside
    ``n_ctx_train`` (the model's native length). The runtime budget must
    respect the serving value, not the larger training figure.
    """
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        note(f"{type(exc).__name__}: {exc}; endpoint not answering at {base_url}")
        return None
    entries = payload["data"]
    entry = next(item for item in entries if item["id"] == model) if model else entries[0]
    n_ctx = entry["context_length"] if "openrouter.ai" in base_url else entry["meta"]["n_ctx"]
    return {"id": entry["id"], "n_ctx": n_ctx}


def wait_for_endpoint(base_url: str, api_key: str, *, model: str | None = None) -> dict[str, Any]:
    """Block until the endpoint answers. Startup must survive a cold server."""
    while True:
        capabilities = probe_endpoint(base_url, api_key, model=model)
        if capabilities is not None:
            return capabilities
        note(f"trying reconnect in {RETRY_DELAY_SECONDS:g}s...")
        time.sleep(RETRY_DELAY_SECONDS)


def build_settings(
    *,
    cwd: str | Path | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    mcp_config: str | Path | None = None,
    shell: str = "auto",
    shell_timeout: int = DEFAULT_SHELL_TIMEOUT,
    enable_thinking: bool = True,
    skill: str | None = None,
    verbose: bool = False,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> Settings:
    """Resolve one runtime configuration, probing the endpoint if needed.

    The model id and the context window are both discoverable from
    ``/v1/models``. When either is left to discovery this blocks until the
    endpoint answers rather than starting a run against a server that is not
    there yet.
    """
    cwd_path = Path(cwd or os.getcwd()).expanduser().resolve()
    if not cwd_path.is_dir():
        raise ValueError(f"working directory does not exist: {cwd_path}")
    resolved_base_url = (
        base_url or env_first("LOCAL_AGENT_BASE_URL", "OPENAI_BASE_URL") or DEFAULT_BASE_URL
    ).rstrip("/")
    resolved_api_key = (
        api_key or env_first("LOCAL_AGENT_API_KEY", "OPENAI_API_KEY") or "local"
    )
    resolved_model = model or env_first("LOCAL_AGENT_MODEL", "OPENAI_MODEL")

    capabilities: dict[str, Any] | None = None
    if resolved_model is None or context_window <= 0:
        capabilities = wait_for_endpoint(resolved_base_url, resolved_api_key, model=resolved_model)
    if resolved_model is None:
        resolved_model = capabilities["id"]
    if context_window <= 0:
        served = capabilities["n_ctx"]
        context_window = served

    backend = select_shell(shell)
    resolved_mcp_config = find_mcp_config(cwd_path, mcp_config)
    return Settings(
        cwd=cwd_path,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        model=resolved_model,
        mcp_config=resolved_mcp_config,
        shell_kind=backend.kind,
        shell_executable=backend.executable,
        shell_timeout=shell_timeout,
        disable_thinking=not enable_thinking,
        skill=skill,
        verbose=verbose,
        context_window=context_window,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local coding agent with interactive and one-shot modes."
    )
    parser.add_argument(
        "--mode",
        choices=("interactive", "auto"),
        default="interactive",
        help="Persistent chat session, or one prompt that runs to completion.",
    )
    parser.add_argument(
        "--auto",
        dest="mode",
        action="store_const",
        const="auto",
        help="Shortcut for --mode auto.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt text in auto mode, or a path to a UTF-8 text file.",
    )
    parser.add_argument("--prompt-file", help="Read the auto-mode prompt from this file.")
    parser.add_argument("--run-id", help="Resume this session directory under --log-root.")
    parser.add_argument("--log-root", default=str(DEFAULT_LOG_ROOT))
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--mcp-config")
    parser.add_argument(
        "--shell",
        choices=("auto", "powershell", "bash"),
        default=os.environ.get("LOCAL_AGENT_SHELL", "auto"),
        help="Host shell. auto selects PowerShell on Windows and Bash elsewhere.",
    )
    parser.add_argument(
        "--shell-timeout",
        type=int,
        default=env_int("LOCAL_AGENT_SHELL_TIMEOUT", DEFAULT_SHELL_TIMEOUT),
        help="Seconds allowed for one host-shell tool call.",
    )
    parser.add_argument(
        "--skill",
        help=(
            "Load exactly one skill and inject its full SKILL.md into the "
            "system prompt, replacing the skill index. Accepts the skill's "
            "name or a path to its SKILL.md."
        ),
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=env_int("LOCAL_AGENT_CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW),
        help=(
            "Token budget used to size compaction. 0 (default) reads the "
            "selected model's advertised serving context length."
        ),
    )
    parser.add_argument(
        "--enable-thinking", dest="enable_thinking", action="store_true", default=True
    )
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the raw model stream to stderr as it arrives.",
    )
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> Settings:
    return build_settings(
        cwd=args.cwd,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        mcp_config=args.mcp_config,
        shell=args.shell,
        shell_timeout=args.shell_timeout,
        enable_thinking=args.enable_thinking,
        skill=args.skill,
        verbose=args.verbose,
        context_window=args.context_window,
    )


# ---------------------------------------------------------------------------
# Host shell
# ---------------------------------------------------------------------------


class ShellBackend(ABC):
    """Only the platform-specific mechanics of the agent's host shell."""

    kind: Literal["powershell", "bash"]
    file_suffix: str
    file_encoding: str = "utf-8"

    def __init__(self, executable: str) -> None:
        self.executable = executable

    @property
    @abstractmethod
    def preamble(self) -> str:
        """Text prepended to every model-proposed script."""

    @abstractmethod
    def invocation(self, script_path: str) -> list[str]:
        """Command used to execute a temporary script file."""


class PowerShellBackend(ShellBackend):
    kind: Literal["powershell"] = "powershell"
    file_suffix = ".ps1"
    file_encoding = "utf-8-sig"

    @property
    def preamble(self) -> str:
        return (
            "$OutputEncoding = [Console]::OutputEncoding = "
            "[System.Text.UTF8Encoding]::new($false)\n"
            "$ProgressPreference = 'SilentlyContinue'\n"
        )

    def invocation(self, script_path: str) -> list[str]:
        return [
            self.executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            script_path,
        ]


class BashBackend(ShellBackend):
    kind: Literal["bash"] = "bash"
    file_suffix = ".sh"

    @property
    def preamble(self) -> str:
        # No `set -e`: the tool reports the script's real exit code and
        # diagnostics instead of changing ordinary shell semantics.
        return "set -o pipefail\n"

    def invocation(self, script_path: str) -> list[str]:
        return [self.executable, "--noprofile", "--norc", script_path]


def select_shell(requested: str = "auto") -> ShellBackend:
    kind = requested
    if kind == "auto":
        kind = "powershell" if os.name == "nt" else "bash"
    if kind == "powershell":
        for executable in ("pwsh.exe", "pwsh", "powershell.exe", "powershell"):
            resolved = shutil.which(executable)
            if resolved:
                return PowerShellBackend(resolved)
        raise RuntimeError("PowerShell was requested but is not on PATH")
    if kind == "bash":
        resolved = shutil.which("bash")
        if resolved:
            return BashBackend(resolved)
        raise RuntimeError("Bash was requested but is not on PATH")
    raise ValueError(f"unsupported shell: {requested}")


def shell_backend(settings: Settings) -> ShellBackend:
    if settings.shell_kind == "powershell":
        return PowerShellBackend(settings.shell_executable)
    return BashBackend(settings.shell_executable)


def _terminate_shell_wrapper(process: subprocess.Popen[bytes]) -> bool:
    """Kill and reap one shell wrapper without introducing another open-ended wait."""
    if process.poll() is not None:
        return True
    with suppress(OSError):
        process.kill()
    try:
        process.wait(timeout=SHELL_TERMINATE_GRACE_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return True


def _shell_output_preview(capture: Any, path: Path) -> str:
    """Read a bounded tail without loading the complete shell output into RAM."""
    capture.seek(0, os.SEEK_END)
    size = capture.tell()
    start = max(0, size - context_limits().shell_chars * 4)
    capture.seek(start)
    text = capture.read(size - start).decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    preview = "".join(lines[-context_limits().shell_lines:])[-context_limits().shell_chars:]
    truncated = start > 0 or preview != text
    notice = (
        f"[output truncated: showing only the tail, at most {context_limits().shell_lines} "
        f"lines / {context_limits().shell_chars} characters]\n"
        if truncated else ""
    )
    return (
        f"{notice}{preview or '(empty)'}\n"
        f"[full output: {path} ({size} bytes captured)]"
    )


def _run_host_shell(
    backend: ShellBackend,
    cwd: Path,
    command: str,
    timeout_seconds: int,
    log_dir: Path | None = None,
) -> str:
    """Run one shell script with a timeout that cannot be held open by descendants.

    Pipes are deliberately not used here. On Windows a background descendant can
    inherit a pipe handle after the shell wrapper exits. ``subprocess.run`` then
    waits for pipe EOF, and its timeout cleanup performs another unbounded
    ``communicate()``. Seekable temporary files let us wait only on the wrapper's
    process handle and read whatever output exists after that bounded wait.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    log_root = log_dir or active_session.path / "shell-output"
    log_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="call_", dir=log_root))
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    script_path: str | None = None
    process: subprocess.Popen[bytes] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=backend.file_suffix,
            prefix="pm_coder_worker_",
            encoding=backend.file_encoding,
            delete=False,
        ) as script_file:
            script_file.write(backend.preamble + command)
            script_path = script_file.name

        with (
            stdout_path.open("w+b") as stdout_capture,
            stderr_path.open("w+b") as stderr_capture,
        ):
            process = subprocess.Popen(
                backend.invocation(script_path),
                cwd=cwd,
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=stdout_capture,
                stderr=stderr_capture,
            )
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                terminated = _terminate_shell_wrapper(process)
                stdout = _shell_output_preview(stdout_capture, stdout_path)
                stderr = _shell_output_preview(stderr_capture, stderr_path)
                detail = "" if terminated else "wrapper_terminated: false\n"
                return (
                    "timed_out: true\n"
                    f"timeout_seconds: {timeout_seconds}\n"
                    f"{detail}"
                    f"stdout_before_timeout:\n{stdout or '(empty)'}\n"
                    f"stderr_before_timeout:\n{stderr or '(empty)'}"
                )

            stdout = _shell_output_preview(stdout_capture, stdout_path)
            stderr = _shell_output_preview(stderr_capture, stderr_path)
            return (
                f"exit_code: {returncode}\n"
                f"stdout:\n{stdout or '(empty)'}\n"
                f"stderr:\n{stderr or '(empty)'}"
            )
    finally:
        if process is not None and process.poll() is None:
            _terminate_shell_wrapper(process)
        if script_path is not None:
            with suppress(OSError):
                os.remove(script_path)


def make_shell_tool(settings: Settings) -> Tool[Any]:
    backend = shell_backend(settings)

    def host_shell(command: str, timeout_seconds: int = settings.shell_timeout) -> str:
        """Execute a host-shell script in the selected agent workspace.

        Complete stdout and stderr are saved to separate persistent log files.
        Each stream returns only its last {shell_lines} lines, capped at
        {shell_chars} characters.
        Truncation is explicit; a tail is not proof that earlier output passed.
        Use read with narrow ranges, or search the reported absolute log paths,
        for missing details. Do not rerun a command just to retrieve its output.
        Prefer focused commands such as git log -5 and targeted searches.
        """
        print(f"\n[{backend.kind}]\n{command.rstrip()}", file=sys.stderr, flush=True)
        result = _run_host_shell(backend, settings.cwd, command, timeout_seconds)
        if result.startswith("timed_out: true"):
            print(f"[{backend.kind} timed out]", file=sys.stderr, flush=True)
        else:
            match = re.match(r"exit_code: (-?\d+)", result)
            print(f"[{backend.kind} exit {match.group(1)}]", file=sys.stderr, flush=True)
        return result

    host_shell.__doc__ = host_shell.__doc__.replace(
        "{shell_lines}", str(context_limits().shell_lines)
    ).replace("{shell_chars}", str(context_limits().shell_chars))

    return Tool(
        host_shell,
        takes_ctx=False,
        name=backend.kind,
        sequential=True,
        max_retries=1,
        strict=False,
    )


def make_bash_machine_tool(machine: Any, user: str) -> Tool[Any]:
    """Shell tool that routes commands into an in-memory BashMachine.

    The BashMachine call is synchronous and serializes on one RLock. No
    subprocess, no timeout, no real filesystem.
    """

    def bash_machine_shell(command: str, timeout_seconds: int = 240) -> str:
        print(
            f"\n[bash-machine:{user}]\n{command.rstrip()}",
            file=sys.stderr,
            flush=True,
        )
        result = machine.exec(user, command)
        print(
            f"[bash-machine:{user} exit {result.exit_code}]",
            file=sys.stderr,
            flush=True,
        )
        return (
            f"exit_code: {result.exit_code}\n"
            f"stdout:\n{result.stdout or '(empty)'}\n"
            f"stderr:\n{result.stderr or '(empty)'}"
        )

    return Tool(
        bash_machine_shell,
        takes_ctx=False,
        name="bash",
        sequential=True,
        max_retries=1,
        strict=False,
    )


# ---------------------------------------------------------------------------
# File tools
#
# read / write / edit exist because doing this through the shell makes the
# model escape the same content twice -- once for the tool-call JSON and again
# for the shell -- and PowerShell is the worst possible second layer. They
# address the file by exact content, never by line number: a wrong string
# fails loudly and can be retried, while a wrong line range succeeds and
# deletes the wrong code, which is not a failure an unattended run survives.
# Line numbers appear only in `read` output, to be quoted back verbatim.
# ---------------------------------------------------------------------------


def resolve_path(settings: Settings, path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = settings.cwd / candidate
    return candidate


def newline_style(text: str) -> str:
    """The line ending a file already uses, so editing it does not convert it."""
    return "\r\n" if "\r\n" in text else "\n"


def split_lines(text: str) -> list[str]:
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def match_hint(text: str, needle: str) -> str:
    """Point at the closest thing in the file to a string that did not match.

    A small model usually misses by one level of indentation or a renamed
    identifier. Naming the nearby lines turns a three-step recovery
    (fail, re-read, retry) into one.
    """
    wanted = needle.strip().splitlines()
    if not wanted:
        return ""
    first = wanted[0].strip()
    scored = sorted(
        (
            (difflib.SequenceMatcher(None, first, line.strip()).ratio(), number, line)
            for number, line in enumerate(split_lines(text), 1)
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    close = [(number, line) for ratio, number, line in scored[:3] if ratio > 0.5]
    if not close:
        return ""
    rendered = "\n".join(f"{number}: {line}" for number, line in close)
    return f"\nClosest lines in the file:\n{rendered}"


def tool_failure(exc: Exception) -> str:
    """Render a tool failure as a normal, machine-readable tool result.

    Tool calls are model input. Bad paths and ranges are expected recovery
    cases, not transport failures that should abort a turn and reconnect.
    """
    message = str(exc) or type(exc).__name__
    return json.dumps({"success": False, "error": message})


def _resolve_read_args(
    start_line: int,
    line_length: int,
    start_column: int,
    column_length: int,
) -> tuple[int, int, int, int]:
    return (
        1 if start_line <= 0 else start_line,
        context_limits().read_lines if line_length <= 0 else line_length,
        1 if start_column <= 0 else start_column,
        context_limits().read_columns if column_length <= 0 else column_length,
    )


def _discard_line_tail(handle) -> tuple[bool, bool]:
    """Consume the rest of one physical line without ever loading it whole.

    Returns (had_more_text, hit_eof).
    """
    had_text = False

    while True:
        chunk = handle.readline(8_192)

        if chunk == "":
            return had_text, True

        if chunk.endswith("\n"):
            return had_text or len(chunk) > 1, False

        had_text = True


def _read_line_slice(
    handle,
    start_column: int,
    length: int,
) -> tuple[str, bool, bool, bool]:
    """Read one bounded slice from the current physical line.

    The handle MUST currently be at the beginning of a physical line.

    Returns:
        text
        has_more_columns
        hit_eof
        line_exists

    The rest of the physical line is consumed before returning, so the handle
    is positioned at the beginning of the next line.
    """
    end_column = start_column + length - 1
    column = 1
    out: list[str] = []
    saw_anything = False

    while True:
        chunk = handle.readline(8_192)

        if chunk == "":
            return "".join(out), False, True, saw_anything

        saw_anything = True

        has_newline = chunk.endswith("\n")
        text = chunk[:-1] if has_newline else chunk

        chunk_start = column
        chunk_end = column + len(text) - 1

        left = max(start_column, chunk_start)
        right = min(end_column, chunk_end)

        if left <= right:
            out.append(text[left - chunk_start : right - chunk_start + 1])

        # We reached the requested column boundary.
        if chunk_end >= end_column:
            if has_newline:
                return "".join(out), chunk_end > end_column, False, True

            tail_has_text, hit_eof = _discard_line_tail(handle)

            return (
                "".join(out),
                chunk_end > end_column or tail_has_text,
                hit_eof,
                True,
            )

        # Physical line ended before the requested column boundary.
        if has_newline:
            return "".join(out), False, False, True

        column = chunk_end + 1


def _bounded_read_stream(
    handle,
    label: str,
    size: int,
    start_line: int,
    line_length: int,
    start_column: int,
    column_length: int,
) -> str:
    start_line, line_length, start_column, column_length = _resolve_read_args(
        start_line,
        line_length,
        start_column,
        column_length,
    )

    if size == 0:
        if start_line != 1:
            raise ValueError(
                f"start_line {start_line} is past the end of an empty file"
            )

        return (
            f"{label} (empty file)\n"
            f"requested start_line={start_line}, line_length={line_length}, "
            f"start_column={start_column}, column_length={column_length}"
        )

    end_column = start_column + column_length - 1

    rendered: list[str] = []
    clipped_lines: list[int] = []
    notes: list[str] = []

    used = 0
    current_line = 1
    processed = 0
    hit_eof = False

    hard_stop_line: int | None = None
    hard_column_stop = False

    # Skip preceding lines in bounded chunks. This remains safe even when
    # one of those physical lines is hundreds of megabytes long.
    while current_line < start_line:
        chunk = handle.readline(8_192)

        if chunk == "":
            raise ValueError(
                f"start_line {start_line} is past the end of the file"
            )

        if chunk.endswith("\n"):
            current_line += 1

    while processed < line_length and not hit_eof:
        prefix = f"{current_line}: "

        # Keep enough reserve that headers / continuation instructions cannot
        # push the complete tool result beyond the hard output fuse.
        remaining = context_limits().read_body_chars - used - len(prefix) - 128

        # Prefer stopping at a clean line boundary instead of returning seven
        # random characters from the next ordinary line.
        if remaining < 512:
            hard_stop_line = current_line
            break

        capture = min(column_length, remaining)

        visible, has_more_columns, hit_eof, exists = _read_line_slice(
            handle,
            start_column,
            capture,
        )

        if not exists:
            if processed == 0:
                raise ValueError(
                    f"start_line {start_line} is past the end of the file"
                )
            break

        # This only happens if the model explicitly requested a column window
        # so enormous that the absolute tool-output fuse was reached.
        hard_column_stop = capture < column_length and has_more_columns

        if hard_column_stop:
            next_column = start_column + len(visible)
            suffix = (
                f" … [hard output limit; line {current_line} "
                f"continues at column {next_column}]"
            )

        elif has_more_columns:
            suffix = (
                f" … [line {current_line} continues at "
                f"column {end_column + 1}]"
            )
            clipped_lines.append(current_line)

        else:
            suffix = ""

        piece = prefix + visible + suffix

        rendered.append(piece)
        used += len(piece) + 1
        processed += 1

        if hard_column_stop:
            notes.append(
                f"HARD OUTPUT LIMIT reached inside line {current_line}. "
                f"Continue with start_line={current_line}, line_length=1, "
                f"start_column={next_column}, "
                f"column_length={column_length}."
            )
            break

        current_line += 1

    if hard_stop_line is not None:
        remaining_lines = max(1, line_length - processed)

        notes.append(
            f"HARD OUTPUT LIMIT reached before line {hard_stop_line}. "
            f"Continue with start_line={hard_stop_line}, "
            f"line_length={remaining_lines}, "
            f"start_column={start_column}, "
            f"column_length={column_length}."
        )

    elif not hard_column_stop and processed >= line_length and not hit_eof:
        # One-character peek only to avoid falsely claiming there is another line.
        if handle.read(1) != "":
            notes.append(
                f"More lines exist. Continue with start_line={current_line}, "
                f"line_length={line_length}, start_column=1, "
                f"column_length={column_length}."
            )

    if clipped_lines:
        shown = ", ".join(str(n) for n in clipped_lines[:20])

        extra = (
            ""
            if len(clipped_lines) <= 20
            else f", ... (+{len(clipped_lines) - 20} more)"
        )

        notes.append(
            f"Long lines clipped at column {end_column}: {shown}{extra}. "
            f"To continue one, use that line with line_length=1, "
            f"start_column={end_column + 1}, "
            f"column_length={column_length}."
        )

    result = (
        f"{label} ({size:,} bytes)\n"
        f"requested start_line={start_line}, line_length={line_length}, "
        f"start_column={start_column}, column_length={column_length}\n"
        + "\n".join(rendered)
    )

    if notes:
        result += "\n\n" + "\n".join(notes)

    # This should be impossible unless somebody later breaks the accounting.
    if len(result) > context_limits().read_output_chars:
        raise RuntimeError(
            f"internal read safety invariant broken: "
            f"{len(result)} > {context_limits().read_output_chars} characters"
        )

    return result


def _replace_line_block(
    raw: str,
    content: str,
    start: int,
    end: int,
) -> str:
    """Return text after a strict whole-file or inclusive line-block write."""

    # WHOLE FILE.
    #
    # Requiring BOTH values to be <= 0 is intentional. If the model sends
    # start=20,end=0 we must not interpret that malformed request as
    # "sure, overwrite the entire file".
    if start <= 0 and end <= 0:
        newline = newline_style(raw) if raw else "\n"
        body = content.replace("\r\n", "\n")
        return body.replace("\n", newline)

    # Mixed whole-file/ranged semantics are always a bug.
    if start <= 0 or end <= 0:
        raise ValueError(
            "invalid write range: start and end must BOTH be <= 0 for a "
            "whole-file write, or BOTH be > 0 for a block replacement"
        )

    if start > end:
        raise ValueError(
            f"invalid write range: start ({start}) is greater than end ({end})"
        )

    newline = newline_style(raw)

    normalized = raw.replace("\r\n", "\n")
    had_final_newline = normalized.endswith("\n")

    lines = split_lines(normalized)

    if start > len(lines) or end > len(lines):
        raise ValueError(
            f"invalid write range {start}-{end}: "
            f"file contains {len(lines)} lines"
        )

    replacement = split_lines(content.replace("\r\n", "\n"))

    # start/end are 1-indexed and INCLUSIVE.
    updated_lines = (
        lines[: start - 1]
        + replacement
        + lines[end:]
    )

    updated = "\n".join(updated_lines)

    # Partial writes preserve whether the existing file ended in a newline.
    if had_final_newline and updated_lines:
        updated += "\n"

    return updated.replace("\n", newline)

IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


# ---------------------------------------------------------------------------
# File backends
#
# read / write / edit are implemented once, below. The only difference
# between the real workspace and an in-memory BashMachine is how bytes get
# on and off a disk, so exactly that syscall-shaped part sits behind this
# small interface and everything above it -- bounded reads, ranged writes,
# exact-text edits, failure results -- is shared verbatim.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostFiles:
    """File operations against the real workspace under ``settings.cwd``."""

    settings: Settings

    def resolve(self, path: str) -> str:
        return str(resolve_path(self.settings, path))

    def is_file(self, target: str) -> bool:
        return Path(target).is_file()

    def size(self, target: str) -> int:
        return Path(target).stat().st_size

    def read_bytes(self, target: str) -> bytes:
        # Bytes, not read_text: universal-newline translation would hide a
        # file's real line endings from newline_style and silently convert
        # every edited CRLF file to LF.
        return Path(target).read_bytes()

    def write_bytes(self, target: str, data: bytes) -> None:
        atomic_write_bytes(Path(target), data)

    def open_text(self, target: str):
        return open(target, "r", encoding="utf-8", errors="replace", newline=None)


@dataclass(frozen=True)
class VirtualFiles:
    """The same operations routed into an in-memory BashMachine."""

    bash_machine: Any

    def resolve(self, path: str) -> str:
        # Virtual paths stay as given; relative ones resolve against the
        # machine's admin cwd (``/home/user``), like its shell commands do.
        return path

    def is_file(self, target: str) -> bool:
        return self.bash_machine.is_file(target)

    def size(self, target: str) -> int:
        return len(self.read_bytes(target))

    def read_bytes(self, target: str) -> bytes:
        return self.bash_machine.read_binary(target)

    def write_bytes(self, target: str, data: bytes) -> None:
        # The file tools only ever write UTF-8 text; storing it as text keeps
        # BashMachine.read_text and shell `cat` working on tool-written files.
        self.bash_machine.write_text(target, data.decode("utf-8"))

    def open_text(self, target: str):
        # The machine already holds the file in memory; the bounded reader
        # streams over a StringIO with the same newline translation the host
        # backend gets from newline=None.
        text = self.read_bytes(target).decode("utf-8", errors="replace")
        return io.StringIO(text.replace("\r\n", "\n").replace("\r", "\n"))


def make_file_tools(settings: Settings, bash_machine: Any = None) -> list[Tool[Any]]:
    """Build read/write/edit against one storage backend.

    With ``bash_machine=None`` the tools operate on the real workspace under
    ``settings.cwd``. With a BashMachine they operate on its in-memory
    filesystem. Signatures and behavior are identical; only the file
    operations behind the tools switch.
    """
    files = VirtualFiles(bash_machine) if bash_machine is not None else HostFiles(settings)

    def read(
            path: str,
            start_line: int = 1,
            line_length: int = context_limits().read_lines,
            start_column: int = 1,
            column_length: int = context_limits().read_columns,
    ):
        """Read the contents of a file. Supports text files and images (jpg, png).
        Images are attached to the conversation so you can see them.

        path: file to read, relative to the working directory or absolute.

        TEXT FILES return a bounded, line-numbered window sized to your context:
            start_line:    1-indexed first line. Default 1 (<=0 means 1).
            line_length:   max lines to return. Default {read_lines} (<=0 means default).
            start_column:  1-indexed first character column inside each line. Default 1.
            column_length: max characters per line. Default {read_columns}.

        Long lines are clipped, never loaded whole. When output stops early the
        result prints exact continuation coordinates -- pass them back in the
        next call to continue where you left off.

        CONTEXT-SAVING RULES:
            - Locate code with the shell first (grep -Rni), then read ONLY the
              lines you need. Never read a whole file to find one function.
            - Do not re-read what you already have in context. Your own
              successful write/edit calls update that knowledge.
            - A full read is read(path, 0, 0, 0, 0). Use it only for files you
              know are small.

        EXAMPLES:
            read("src/foo.py")                       first {read_lines} lines
            read("src/foo.py", 500, 100)             lines 500-599
            read("bundle.min.js", 1, 1, 4001, 4000)  next 4000 chars of a huge one-line file
            read("diagram.png")                      attach the image itself
        """
        try:
            target = files.resolve(path)
            print(
                f"\n[read {target} start_line={start_line} line_length={line_length} "
                f"start_column={start_column} column_length={column_length}]",
                file=sys.stderr,
                flush=True,
            )
            if not files.is_file(target):
                raise FileNotFoundError(f"file does not exist: {target}")

            media_type = IMAGE_MEDIA_TYPES.get(Path(str(target)).suffix.lower())
            if media_type is not None:
                # Pydantic AI turns a BinaryContent tool return into a base64
                # image_url user message; the framework does the rest. If the
                # endpoint has no projector, run_turn fails loudly on its 500.
                data = files.read_bytes(target)
                if not data:
                    raise ValueError(f"file is empty: {target}")
                return [
                    BinaryContent(data, media_type=media_type),
                    f"image attached: {target} ({len(data):,} bytes)",
                ]

            if (
                "skill" in str(target).casefold()
                and (start_line <= 0 or start_line == 1)
                and (line_length <= 0 or line_length == context_limits().read_lines)
                and (start_column <= 0 or start_column == 1)
                and (column_length <= 0 or column_length == context_limits().read_columns)
            ):
                line_length = sys.maxsize
                column_length = sys.maxsize

            with files.open_text(target) as handle:
                return _bounded_read_stream(
                    handle,
                    str(target),
                    files.size(target),
                    start_line,
                    line_length,
                    start_column,
                    column_length,
                )
        except Exception as exc:
            return tool_failure(exc)

    def write(
            path: str,
            content: str,
            start: int = 0,
            end: int = 0,
    ) -> str:
        """Write content to a file. Creates the file if it doesn't exist,
        overwrites if it does. Automatically creates parent directories.

        USE THIS FOR:
            - a new file
            - a deliberate full rewrite
            Use edit() to CHANGE existing content: it fails loudly when the
            old text does not match, where a rewrite silently destroys work.

        WHOLE-FILE WRITE (the default: start=0, end=0):
            write("src/foo.py", content)

        BLOCK REPLACEMENT (start > 0 AND end > 0, 1-indexed, INCLUSIVE):
            write("src/foo.py", replacement, 20, 35)
            Replaces lines 20 THROUGH 35. content="" deletes those lines.
            Use it to continue a large file in chunks of at most
            {write_chunk_chars} characters.

        INVALID requests return an error instead of guessing: mixed ranges
            (start <= 0, end > 0), start > end, ranges outside the file,
            ranged writes to a file that does not exist.

        Line numbers come from read output. If an edit or external change
        made them uncertain, re-read only the affected range -- not the file.
        """
        try:
            target = files.resolve(path)
            print(
                f"\n[write {target} start={start} end={end}]",
                file=sys.stderr,
                flush=True,
            )
            whole_file = start <= 0 and end <= 0

            if not whole_file and not files.is_file(target):
                raise FileNotFoundError(
                    f"cannot perform ranged write: file does not exist: {target}"
                )

            raw = (
                files.read_bytes(target).decode("utf-8", errors="replace")
                if files.is_file(target)
                else ""
            )

            updated = _replace_line_block(
                raw,
                content,
                start,
                end,
            )

            files.write_bytes(target, updated.encode("utf-8"))
        except Exception as exc:
            return tool_failure(exc)

        mode = (
            "whole file"
            if whole_file
            else f"lines {start}-{end}"
        )

        return (
            f"wrote {target} ({mode}; "
            f"file is now {len(split_lines(updated))} lines)"
        )

    def edit(
        path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Replace exact text in a file. The preferred way to change existing files.

        old_string must match the file EXACTLY, including whitespace,
        indentation, and newlines -- copy it verbatim from read output. It
        must match exactly once unless replace_all=true.

        A failed match is cheap: the error names the closest matching lines,
        so fix old_string and retry instead of re-reading the whole file.

        RULES:
            - Prefer one larger edit of a coherent block over several small
              interleaved ones.
            - Keep old_string as small as possible while still unique.
            - To create a file use write; old_string must not be empty.
            - The file's line endings (CRLF or LF) are preserved.
        """
        target = files.resolve(path)
        print(f"\n[edit {target}]", file=sys.stderr, flush=True)
        if not files.is_file(target):
            return f"error: file does not exist: {target}"
        raw = files.read_bytes(target).decode("utf-8", errors="replace")
        newline = newline_style(raw)
        text = raw.replace("\r\n", "\n")
        old = old_string.replace("\r\n", "\n")
        new = new_string.replace("\r\n", "\n")
        if not old:
            return "error: old_string is empty; use write to create a file"
        count = text.count(old)
        if count == 0:
            return (
                f"error: no match for old_string in {target}. It must match the "
                "file exactly, including whitespace and indentation."
                + match_hint(text, old)
            )
        if count > 1 and not replace_all:
            return (
                f"error: found {count} matches for old_string in {target}. Add "
                "surrounding context to make it unique, or pass replace_all=true."
            )
        line = text[: text.index(old)].count("\n") + 1
        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        files.write_bytes(target, updated.replace("\n", newline).encode("utf-8"))
        where = f"{count} occurrences" if replace_all else f"line {line}"
        return f"edited {target} ({where}, file is now {len(split_lines(updated))} lines)"

    # These must be ordinary docstrings for the tool schema. Substitute the
    # limits after definition because an f-string would not be a docstring.
    assert read.__doc__ is not None
    read.__doc__ = (
        read.__doc__
        .replace("{read_lines}", str(context_limits().read_lines))
        .replace("{read_columns}", str(context_limits().read_columns))
    )
    assert write.__doc__ is not None
    write.__doc__ = (
        write.__doc__
        .replace("{write_chunk_chars}", str(context_limits().write_chunk_chars))
    )

    return [
        Tool(read, takes_ctx=False, name="read", sequential=True, strict=False),
        Tool(write, takes_ctx=False, name="write", sequential=True, strict=False),
        Tool(edit, takes_ctx=False, name="edit", sequential=True, strict=False),
    ]


# ---------------------------------------------------------------------------
# Workspace discovery: MCP config, skills, project instructions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    skill_file: Path
    priority: int


@dataclass(frozen=True)
class DiscoveryResult:
    skills: list[Skill]
    skill_errors: list[str]
    instruction_files: list[Path]
    project_instructions: str
    mcp_server_names: list[str]
    selected_skill: Skill | None


def ancestors_nearest_first(cwd: Path) -> list[Path]:
    directories: list[Path] = []
    current = cwd.resolve()
    while True:
        directories.append(current)
        if current.parent == current:
            return directories
        current = current.parent


def find_mcp_config(cwd: Path, explicit: str | Path | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = cwd / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MCP config does not exist: {path}")
        return path
    for directory in ancestors_nearest_first(cwd):
        for name in MCP_CONFIG_CANDIDATES:
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()
    return None


def read_mcp_server_names(config_path: Path | None) -> list[str]:
    if config_path is None:
        return []
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return [str(name) for name in payload["mcpServers"]]


def skill_roots(cwd: Path) -> list[Path]:
    roots: list[Path] = []
    for directory in ancestors_nearest_first(cwd):
        roots += [
            directory / ".agents" / "skills",
            directory / ".pi" / "skills",
            directory / ".codex" / "skills",
        ]
    roots += [
        Path.home() / ".agents" / "skills",
        Path.home() / ".pi" / "agent" / "skills",
        Path.home() / ".codex" / "skills",
        Path.home() / ".pm" / "skills",
    ]
    unique: list[Path] = []
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def fallback_skill_description(body: str, skill_name: str) -> str:
    for paragraph in re.split(r"\r?\n\s*\r?\n", body):
        lines: list[str] = []
        for raw_line in paragraph.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("```"):
                continue
            if line.startswith("#"):
                line = line.lstrip("#").strip()
                if line.casefold() == skill_name.casefold():
                    continue
            lines.append(line)
        if lines:
            return " ".join(lines)
    return f"Reusable workflow from {skill_name}."


def parse_skill(skill_file: Path, priority: int) -> Skill:
    text = skill_file.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    metadata: dict[str, Any] = {}
    body = text
    if match:
        metadata = load_yaml(match.group(1)) or {}
        body = text[match.end() :]
    name = str(metadata.get("name") or skill_file.parent.name).strip()
    raw_description = metadata.get("description")
    description = (
        str(raw_description).strip()
        if raw_description is not None
        else fallback_skill_description(body, name)
    )
    description = re.sub(r"\s+", " ", description).strip()
    if not name or not description:
        raise ValueError("skill name and description must not be blank")
    return Skill(name, description[:800], skill_file.resolve(), priority)


def load_skills(cwd: Path) -> tuple[list[Skill], list[str]]:
    skills_by_name: dict[str, Skill] = {}
    seen_files: set[Path] = set()
    errors: list[str] = []
    for priority, root in enumerate(skill_roots(cwd)):
        if not root.is_dir():
            continue
        for skill_file in sorted(root.rglob("SKILL.md")):
            resolved = skill_file.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            try:
                skill = parse_skill(resolved, priority)
            except Exception as exc:
                errors.append(f"{resolved}: {exc}")
                continue
            skills_by_name.setdefault(skill.name.casefold(), skill)
    ordered = sorted(
        skills_by_name.values(), key=lambda skill: (skill.priority, skill.name.casefold())
    )
    return ordered, errors


def format_skill_index(skills: list[Skill]) -> str:
    if not skills:
        return "<available_skills />"
    entries = "".join(
        "  <skill>\n"
        f"    <name>{escape(skill.name)}</name>\n"
        f"    <description>{escape(skill.description)}</description>\n"
        f"    <location>{escape(str(skill.skill_file))}</location>\n"
        "  </skill>\n"
        for skill in skills
    )
    return f"<available_skills>\n{entries}</available_skills>"


def discover_instruction_files(cwd: Path) -> list[Path]:
    files: list[Path] = []
    for directory in reversed(ancestors_nearest_first(cwd)):
        for filename in ("AGENTS.md", "CLAUDE.md"):
            candidate = directory / filename
            if candidate.is_file() and candidate.resolve() not in files:
                files.append(candidate.resolve())
    return files


def load_project_instructions(files: list[Path]) -> str:
    if not files:
        return "<project_instructions />"
    entries = "".join(
        f"  <instruction_file path={quoteattr(str(path))}>\n"
        f"{path.read_text(encoding='utf-8')}\n"
        "  </instruction_file>\n"
        for path in files
    )
    return f"<project_instructions>\n{entries}</project_instructions>"


def find_skill(skills: list[Skill], reference: str) -> Skill:
    """Resolve --skill by name, or by a path to a SKILL.md the caller already resolved."""
    wanted = reference.casefold()
    for skill in skills:
        if skill.name.casefold() == wanted:
            return skill
    candidate = Path(reference).expanduser()
    if candidate.is_file() and candidate.name == "SKILL.md":
        resolved = candidate.resolve()
        for skill in skills:
            if skill.skill_file == resolved:
                return skill
        return parse_skill(resolved, priority=0)
    available = ", ".join(sorted(skill.name for skill in skills)) or "(none)"
    raise FileNotFoundError(
        f"Skill {reference!r} was not found. Available skills: {available}"
    )


def discover_workspace(settings: Settings) -> DiscoveryResult:
    skills, skill_errors = load_skills(settings.cwd)
    selected_skill = find_skill(skills, settings.skill) if settings.skill else None
    instruction_files = discover_instruction_files(settings.cwd)
    return DiscoveryResult(
        skills=skills,
        skill_errors=skill_errors,
        instruction_files=instruction_files,
        project_instructions=load_project_instructions(instruction_files),
        mcp_server_names=read_mcp_server_names(settings.mcp_config),
        selected_skill=selected_skill,
    )


def build_system_prompt(
    settings: Settings,
    discovery: DiscoveryResult,
    *,
    shell_kind_override: str | None = None,
    shell_executable_override: str | None = None,
) -> str:
    shell = shell_backend(settings)
    kind = shell_kind_override or shell.kind
    executable = shell_executable_override or shell.executable
    bash_machine_note = (
        "This is an in-memory Bash environment. There is no network, "
        "no real filesystem, and no background processes. "
        "File paths are virtual (e.g. /home/user/notes.txt). "
        "Use `read`, `write`, and `edit` for all file access. "
        "Multiple agents may share this environment and work on "
        "different files at the same time."
    ) if shell_kind_override == "bash-machine" else ""
    mcp_summary = ", ".join(discovery.mcp_server_names) or "none configured"
    if any("minecraft" in name.casefold() for name in discovery.mcp_server_names):
        mcp_guidance = (
            "MCP tools are exposed with server prefixes. Observe Minecraft "
            "before acting and use ordinary survival mechanics only. A process "
            "exit is not proof of game success."
        )
    else:
        mcp_guidance = (
            "MCP tools are exposed with server prefixes. Use them when the "
            "assigned workflow or project instructions require them."
        )
    if discovery.selected_skill is not None:
        body = discovery.selected_skill.skill_file.read_text(encoding="utf-8")
        skill_block = (
            "A specific skill was selected on the command line. Read and follow "
            "it for the entire session.\n\n"
            f"<selected_skill name={quoteattr(discovery.selected_skill.name)}>\n"
            f"{body}\n"
            "</selected_skill>"
        )
    else:
        skill_block = format_skill_index(discovery.skills)
    return textwrap.dedent(
        f"""
        You are a local coding agent "pm-coder" operating directly in one agent workspace. 
        Your context size is {settings.context_window}.

        <environment>
          <working_directory>{escape(str(settings.cwd))}</working_directory>
          <host_shell name={quoteattr(kind)}>
            {escape(executable)}
          </host_shell>
          <mcp_servers>{escape(mcp_summary)}</mcp_servers>
        </environment>

        Complete the user's current request. Inspect real state, make concrete
        progress, verify every claimed effect, and leave durable evidence when
        useful. Conversation history may continue across requests, so use it as
        context without repeating completed work.

        Conserve tokens. Reuse file contents, discovered facts, and check results
        already present in your context or checkpoint. Do not repeat repository
        reconnaissance after compaction. Your own successful edits update that
        knowledge; they do not require reading the entire file again. Re-read
        only a narrow missing detail, or content affected by evidence of an
        external change, an uncertain edit outcome, or an exact-text mismatch.
        Do not poll files or rerun unchanged baseline checks merely to reassure
        yourself. Run relevant checks after changes to verify their effects.
        Provider KV caching is an optimization, not extra memory: use retained
        context and checkpoint facts, not an assumed cache of omitted content.

        Make one tool call per response and wait for its result before
        deciding the next one. Tool calls run strictly in order anyway, so
        batching several into one response buys nothing and costs you the
        chance to react to what each one returned.

        Use `read`, `write`, and `edit` for files. `read` also returns JPG
        and PNG images as visual attachments, so looking at a picture is
        just reading it. `edit` replaces an exact, unique string: copy
        `old_string` verbatim out of a `read`, including its indentation.
        Use `write` for a new file or a deliberate full rewrite. Prefer one
        larger `edit` of a coherent block over several small interleaved ones.

        `{kind}` starts in the selected workspace. Use it for everything
        else: running commands, searching, and verifying results. Project
        instructions define the exact writable paths. Do not escape the
        selected workspace.

        You may read the absolute output-log paths reported by shell tools,
        including logs outside the workspace. Search or read a narrow range
        when output was truncated; do not reload the complete log into context.

        {bash_machine_note}

        {mcp_guidance}

        Skills are reusable workflows. If one clearly applies, read its full
        SKILL.md before using it.

        {skill_block}

        Apply these discovered project instructions:

        {discovery.project_instructions}
        """
    ).strip()


# ---------------------------------------------------------------------------
# Verbose mode (-v)
#
# VerboseModel wraps the model so the raw response stream reaches stderr while
# it happens, before any of it is validated, retried, or persisted. Wrapping
# at the model means every caller gets it: the main agent and the summarizer.
# ---------------------------------------------------------------------------


class VerbosePrinter:
    """Prints one raw model response to a stream while it streams in."""

    def __init__(self, stream: Any, label: str) -> None:
        self.stream = stream
        self.label = label
        self._raw_active = False  # raw content written without a trailing \n
        self.saw_part = False  # at least one part_start event observed

    def _line(self, text: str) -> None:
        if self._raw_active:
            print("", file=self.stream, flush=True)
            self._raw_active = False
        print(f"{APP_NAME} {self.label}: {text}", file=self.stream, flush=True)

    def _raw(self, value: Any) -> None:
        """Print a part's content verbatim so it reads as the model emitted it."""
        if not value:
            return
        if isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        self._raw_active = True
        self.stream.write(str(value))
        self.stream.flush()

    def request_banner(self, model_name: str, messages: int, tools: int) -> None:
        self._line(f"request: model={model_name} messages={messages} tools={tools}")

    def _part(self, part: Any) -> None:
        kind = getattr(part, "part_kind", "") or "unknown"
        if kind == "text":
            self._line("assistant text:")
            self._raw(getattr(part, "content", None))
        elif kind == "thinking":
            self._line("assistant thinking:")
            self._raw(getattr(part, "content", None))
        elif kind in {"tool-call", "builtin-tool-call"}:
            self._line(f"tool call: {getattr(part, 'tool_name', '') or '?'}")
            self._raw(getattr(part, "args", None))
        else:
            self._line(f"{kind} part")

    def handle(self, event: Any) -> None:
        kind = getattr(event, "event_kind", None)
        if kind == "part_start":
            self.saw_part = True
            self._part(event.part)
        elif kind == "part_delta":
            delta = event.delta
            if isinstance(delta, TextPartDelta | ThinkingPartDelta):
                self._raw(delta.content_delta)
            elif isinstance(delta, ToolCallPartDelta):
                self._raw(delta.tool_name_delta)
                self._raw(delta.args_delta)
        elif kind == "part_end":
            if self._raw_active:
                print("", file=self.stream, flush=True)
                self._raw_active = False

    def print_parts(self, response: Any) -> None:
        """Render final parts when the stream surfaced no part events at all."""
        for part in getattr(response, "parts", None) or []:
            self._part(part)

    def response_done(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        usage_text = (
            f" usage: input={usage.input_tokens} output={usage.output_tokens}"
            if usage is not None
            else ""
        )
        self._line(
            f"response complete: "
            f"finish_reason={getattr(response, 'finish_reason', None) or '?'}{usage_text}"
        )

    def stream_error(self, exc: BaseException) -> None:
        self._line(f"stream error: {type(exc).__name__}: {str(exc)[:400]}")


class VerboseStreamedResponse(StreamedResponse):
    """Pass-through stream that prints every raw event before yielding it.

    Events reach Pydantic AI untouched, and every accessor delegates to the
    inner stream, so the agent loop and usage accounting see exactly the state
    they would without verbose mode.
    """

    def __init__(self, inner: StreamedResponse, printer: VerbosePrinter) -> None:
        super().__init__(inner.model_request_parameters)
        self._inner = inner
        self._printer = printer

    def __aiter__(self) -> Any:
        inner = self._inner
        printer = self._printer

        async def tee() -> Any:
            try:
                async for event in inner:
                    printer.handle(event)
                    if event.event_kind == "final_result":
                        self.final_result_event = event
                    yield event
                if not printer.saw_part:
                    printer.print_parts(inner.get())
                printer.response_done(inner.get())
            except BaseException as exc:
                printer.stream_error(exc)
                raise

        return tee()

    async def _get_event_iterator(self) -> Any:
        # Never used: __aiter__ above consumes the inner stream directly.
        raise NotImplementedError

    async def close_stream(self) -> None:
        await self._inner.close_stream()

    def get(self) -> Any:
        return self._inner.get()

    def time_to_first_chunk(self, request_start: float) -> float | None:
        return self._inner.time_to_first_chunk(request_start)

    @property
    def usage(self) -> Any:
        return self._inner.usage

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def provider_name(self) -> str | None:
        return self._inner.provider_name

    @property
    def provider_url(self) -> str | None:
        return self._inner.provider_url

    @property
    def timestamp(self) -> datetime:
        return self._inner.timestamp


class VerboseModel(WrapperModel):
    """A WrapperModel that tees raw model output to stderr."""

    def __init__(self, wrapped: Any, label: str) -> None:
        super().__init__(wrapped)
        # `label` is a read-only property on Model, so use a distinct name.
        self.verbose_label = label

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[Any],
        model_settings: Any,
        model_request_parameters: Any,
        run_context: Any = None,
    ) -> Any:
        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as inner:
            printer = VerbosePrinter(sys.stderr, self.verbose_label)
            printer.request_banner(
                inner.model_name,
                len(messages),
                len(model_request_parameters.function_tools),
            )
            yield VerboseStreamedResponse(inner, printer)

    async def request(
        self, messages: list[Any], model_settings: Any, model_request_parameters: Any
    ) -> Any:
        response = await self.wrapped.request(
            messages, model_settings, model_request_parameters
        )
        printer = VerbosePrinter(sys.stderr, self.verbose_label)
        printer.request_banner(
            getattr(response, "model_name", "?"),
            len(messages),
            len(model_request_parameters.function_tools),
        )
        printer.print_parts(response)
        printer.response_done(response)
        return response


def _stream_json_default(value: Any) -> Any:
    """Make Pydantic AI stream events readable without depending on internals."""
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


class StreamSpoolResponse(StreamedResponse):
    """Tee one main-agent response into its rolling on-disk recovery spool."""

    def __init__(self, inner: StreamedResponse, handle: Any) -> None:
        super().__init__(inner.model_request_parameters)
        self._inner = inner
        self._handle = handle

    def _record(self, value: dict[str, Any]) -> None:
        self._handle.write(
            json.dumps(value, ensure_ascii=False, default=_stream_json_default) + "\n"
        )
        self._handle.flush()

    def __aiter__(self) -> Any:
        inner = self._inner

        async def tee() -> Any:
            try:
                async for event in inner:
                    self._record(
                        {
                            "event_kind": getattr(event, "event_kind", None),
                            "event_type": type(event).__name__,
                            "index": getattr(event, "index", None),
                            "part": getattr(event, "part", None),
                            "delta": getattr(event, "delta", None),
                        }
                    )
                    if event.event_kind == "final_result":
                        self.final_result_event = event
                    yield event
            except BaseException as exc:
                self._record(
                    {"stream_error": f"{type(exc).__name__}: {exc}"}
                )
                raise
            finally:
                self._handle.close()

        return tee()

    async def _get_event_iterator(self) -> Any:
        raise NotImplementedError

    async def close_stream(self) -> None:
        await self._inner.close_stream()

    def get(self) -> Any:
        return self._inner.get()

    def time_to_first_chunk(self, request_start: float) -> float | None:
        return self._inner.time_to_first_chunk(request_start)

    @property
    def usage(self) -> Any:
        return self._inner.usage

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def provider_name(self) -> str | None:
        return self._inner.provider_name

    @property
    def provider_url(self) -> str | None:
        return self._inner.provider_url

    @property
    def timestamp(self) -> datetime:
        return self._inner.timestamp


class StreamSpoolModel(WrapperModel):
    """Capture only the main agent's current response stream for recovery."""

    def __init__(self, wrapped: Any, session: SessionStore) -> None:
        super().__init__(wrapped)
        self._session = session

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[Any],
        model_settings: Any,
        model_request_parameters: Any,
        run_context: Any = None,
    ) -> Any:
        handle = self._session.begin_stream_capture()
        try:
            async with self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as inner:
                yield StreamSpoolResponse(inner, handle)
        finally:
            if not handle.closed:
                handle.close()


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def make_model(
    settings: Settings, label: str, stream_session: SessionStore | None = None
) -> Any:
    model: Any = LoggingOpenAIChatModel(
        settings.model,
        provider=OpenAIProvider(base_url=settings.base_url, api_key=settings.api_key),
        profile=OpenAIModelProfile(
            openai_supports_strict_tool_definition=False,
            openai_chat_supports_multiple_system_messages=False,
        ),
    )
    if settings.verbose:
        model = VerboseModel(model, label)
    if stream_session is not None:
        model = StreamSpoolModel(model, stream_session)
    return model


def thinking_body(settings: Settings) -> dict[str, Any]:
    if settings.disable_thinking:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


@dataclass
class SnapshotToolset(WrapperToolset[Any]):
    """Persist the in-progress history after a tool call.

    A single ``agent.run`` driving a Minecraft bot can run for hours, and
    nothing reached disk until it finished or failed -- so a crash at hour
    nine lost all nine hours. A completed tool call is the natural checkpoint:
    it happens often and the history is consistent right after one. This wraps
    every toolset, MCP servers included. The throttle lives on the session,
    not here, because Pydantic AI rebuilds toolsets per run and per step.
    """

    session: SessionStore | None = None

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: Any, tool: Any
    ) -> Any:
        result = await self.wrapped.call_tool(name, tool_args, ctx, tool)
        # Only after the call succeeded, and never in a way that can fail the
        # tool: a broken snapshot must not take the run down with it.
        try:
            self.session.snapshot()
        except Exception as exc:
            note(f"snapshot failed: {exc!r}")
        self.session.record_tool_call(name, tool_args)
        return result


def build_agent(
    settings: Settings,
    discovery: DiscoveryResult,
    session: SessionStore,
    *,
    bash_machine: Any = None,
    bash_machine_user: str = "user",
    extra_instructions: str = "",
    with_subagent_tool: bool = False,
    capture_stream: bool = False,
) -> Agent[Any, str]:
    shell_tool = (
        make_bash_machine_tool(bash_machine, bash_machine_user)
        if bash_machine is not None
        else make_shell_tool(settings)
    )
    file_tools = make_file_tools(settings, bash_machine)
    system_prompt = build_system_prompt(
        settings,
        discovery,
        shell_kind_override="bash-machine" if bash_machine is not None else None,
        shell_executable_override=(
            f"in-memory BashMachine, virtual user {bash_machine_user!r}"
            if bash_machine is not None
            else None
        ),
    )
    own_tools = FunctionToolset(
        tools=[shell_tool, *file_tools]
        + (
            [
                make_subagent_tool(
                    settings,
                    bash_machine=bash_machine,
                    bash_machine_user=bash_machine_user,
                )
            ]
            if with_subagent_tool
            else []
        )
    )
    mcp_tools = (
        load_mcp_toolsets(settings.mcp_config) if settings.mcp_config is not None else []
    )
    toolset = SnapshotToolset(
        CombinedToolset([own_tools, *mcp_tools]),
        session=session,
    )
    return Agent(
        model=make_model(settings, "agent", session if capture_stream else None),
        instructions=system_prompt + extra_instructions,
        toolsets=[toolset],
        model_settings=OpenAIChatModelSettings(
            max_tokens=settings.context_window, # stupid fucking ass setting, keep at max and let autocompact handle it
            parallel_tool_calls=False,
            extra_body=thinking_body(settings),
        ),
        capabilities=[
            ProcessHistory(keep_recent_images),
            ProcessHistory(loop_alert_injector(session)),
        ],
        retries=3,
        max_concurrency=1,
    )


def build_summary_agent(settings: Settings) -> Agent[Any, str]:
    """A tool-less agent that turns conversation prefixes into checkpoints."""
    return Agent(
        model=make_model(settings, "compact"),
        system_prompt=(
            "You summarize a coding-agent conversation into a concise, "
            "structured checkpoint that another LLM will use to continue the "
            "work. Preserve exact file paths, function names, and error "
            "messages. Return only the summary."
        ),
        model_settings=OpenAIChatModelSettings(
            temperature=0.0,
            max_tokens=settings.context_window,
            parallel_tool_calls=False,
            extra_body=thinking_body(settings),
        ),
        retries=1,
    )


# ---------------------------------------------------------------------------
# Compaction: triggered only by endpoint rejection or truncated generation.
# ---------------------------------------------------------------------------

SUMMARIZATION_PROMPT = """
You are compacting the execution history of an autonomous coding agent.

Produce a dense checkpoint containing only information useful for continuing
the task correctly.

Preserve:
- the user's requirements, constraints, corrections, and acceptance criteria
- important facts discovered about the project
- architectural or implementation decisions that still matter
- files/symbols that were changed and what was changed
- tests/checks already run and their results
- important errors and their causes
- failed approaches worth not repeating
- unfinished work, open questions, and the next useful actions

Do NOT preserve:
- chain of thought, reasoning narration, or speculation that led nowhere
- conversational filler
- chronological narration merely for completeness
- full file contents, diffs, directory listings, command output, or other
  information that can cheaply be obtained again from the filesystem, git,
  or tools

The filesystem, repository, git working tree, and tools are persistent external
memory. Preserve conclusions from completed reads and checks, with exact paths
and symbols. Do not turn completed reconnaissance into a list of files to read
again. Name a specific next implementation or test action and only the narrow
reads it requires. Later verified facts supersede earlier conflicting claims.

Distinguish facts from unresolved hypotheses when that matters.
Do not invent anything.

Write one consolidated checkpoint as compact Markdown, at most {summary_chars} characters.
Merge earlier checkpoints into current state; do not stack historical summaries.
"""

FAKE_USER_RESUME = "/resume"

CONTEXT_RECOVERY_PROMPT = (
    "The earlier conversation reached the model context window and was "
    "summarized into the checkpoint above. Continue the CURRENT task from that "
    "checkpoint. Do NOT restart reconnaissance or repeat completed reads and "
    "baseline checks. Trust retained findings unless there is evidence of a "
    "change or contradiction. Take the next unfinished implementation or test "
    "action; read only specific missing details needed for it. Verify your "
    "changes and return the required result when done."
)

CONTEXT_MARKERS = (
    "token limit",
    "context limit",
    "context window",
    "context length",
    "context size",
    "context full",
    "maximum context",
    "exceeds the context",
    "exceeded the context",
    "exceeds the available context",
    "available context",
    "prompt is too long",
    "too many tokens",
    "reduce prompt",
    "exceeded",
    "max_tokens",
    "max-tokens",
    "n_ctx",
)

# Errors that mean the endpoint rejected what the model *produced*, not what we
# sent. llama.cpp answers 500 when the tool call the model emitted is not
# parsable JSON; that happens at any context size, while a genuine out-of-room
# rejection is a 4xx carrying one of the markers above. Reading it as "no room"
# cost a session its history once: a 4,798-token conversation was compacted to
# 545 tokens, because the forced path skips the size check.
GENERATION_FAILURE_MARKERS = ("failed to parse", "as json",)

# The endpoint rejecting what *we* attached: llama.cpp without --mmproj
# answers 500 "image input is not supported". The image sits in the history,
# so the plain reconnect-and-retry path resubmits it forever.
UNSUPPORTED_INPUT_MARKERS = ("image input is not supported", "mmproj",)


def error_text(exc: BaseException) -> str:
    """Flattened message plus response body, folded, for marker matching."""
    text = " ".join(str(exc).split()).casefold()
    body = getattr(exc, "body", None)
    if body is not None:
        text += " " + json.dumps(body, default=str).casefold()
    return text


def is_context_failure(exc: BaseException) -> bool:
    """True when an exception means "no room left", however it was phrased."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status >= 500:
        # A server fault is never the endpoint saying the prompt is too long --
        # that rejection is a 4xx -- and this answer forces a compaction past
        # every size check, so a 500 must not be able to reach it.
        return False
    return any(marker in error_text(exc) for marker in CONTEXT_MARKERS)


def is_generation_failure(exc: BaseException) -> bool:
    """True when the model's own output was unusable, whatever the room left."""
    return any(marker in error_text(exc) for marker in GENERATION_FAILURE_MARKERS)


def is_unsupported_input(exc: BaseException) -> bool:
    """True when the endpoint rejects a capability the request uses (vision)."""
    return any(marker in error_text(exc) for marker in UNSUPPORTED_INPUT_MARKERS)


# pydantic-ai feeds a tool's validation error back to the model as a retry
# prompt, and raises UnexpectedModelBehavior once the same tool has failed its
# `retries` times in a row -- the model read the feedback and still emitted
# the same bad arguments. Both phrasings below are pydantic-ai's own; unlike
# urllib3's "max retries exceeded with url", they are about the model's
# output, not the transport.
RETRY_EXHAUSTION_MARKERS = (
    "exceeded max retries count",       # tool arguments failed validation
    "exceeded maximum output retries",   # output validators rejected the reply
)


def is_retry_exhaustion(exc: BaseException) -> bool:
    """True when the model burned every chance to repair its own output."""
    return any(marker in error_text(exc) for marker in RETRY_EXHAUSTION_MARKERS)


def is_model_intelligence_failure(exc: BaseException) -> bool:
    """True when the model is stuck on output it cannot fix itself.

    Two shapes, one cause: a generation the endpoint could not parse, or tool
    arguments the model kept emitting despite the validation error being fed
    back to it. Both are probability traps -- resampling the same history at
    low temperature reproduces the same mistake -- so both want the same cure:
    a fresh user turn to shift the distribution, never a retry as-is.
    """
    return is_generation_failure(exc) or is_retry_exhaustion(exc)


def has_unanswered_tool_call(message: Any) -> bool:
    return isinstance(message, ModelResponse) and any(
        type(part).__name__ == "ToolCallPart" for part in message.parts
    )


def is_unprocessed_tool_calls_error(exc: Exception) -> bool:
    """Exactly pydantic-ai's refusal to take a prompt over a dangling tool call."""
    return "unprocessed tool calls" in str(exc)


def drop_last_tool_call(history: list[Any]) -> list[Any]:
    """Remove the newest tool call from the last assistant message with one.

    The message itself is kept -- its text is real work -- unless nothing
    but the tool call remains, in which case it goes too.
    """
    history = list(history)
    for i in range(len(history) - 1, -1, -1):
        message = history[i]
        if isinstance(message, ModelResponse) and any(
            type(part).__name__ == "ToolCallPart" for part in message.parts
        ):
            kept = [
                part for part in message.parts if type(part).__name__ != "ToolCallPart"
            ]
            if kept:
                history[i] = replace(message, parts=kept)
            else:
                history.pop(i)
            break
    return history


def drop_unanswered_tail(history: list[Any]) -> list[Any]:
    """Remove a trailing response that ends in a tool call nobody answered.

    A truncated response that only produced text is worth keeping: it is real
    work, and the model can carry on from it. One that stopped inside a tool
    call is not -- its arguments are incomplete, and providers reject a history
    where a tool call is followed by anything but its result.
    """
    history = list(history)
    while history and has_unanswered_tool_call(history[-1]):
        history.pop()
    return history


def hit_generation_limit(history: list[Any]) -> bool:
    """True when the newest response stopped because it ran out of room.

    This is the reliable signal for a truncated generation, and it works
    whether the run returned normally or blew up on the way out. Pydantic AI
    raises rather than returns when a truncated response is unusable -- empty,
    thinking-only, or ending in a tool call with unparsable arguments -- and
    each of those messages is worded differently, but all of them leave the
    same ``finish_reason`` on the captured response.
    """
    for message in reversed(history):
        if isinstance(message, ModelResponse):
            return message.finish_reason == "length"
    return False


def content_text(part: dict[str, Any]) -> str:
    content = part.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered: list[str] = []
        for block in content:
            if isinstance(block, str):
                rendered.append(block)
            elif isinstance(block, dict):
                if block.get("kind") == "binary":
                    rendered.append(f"[{block.get('media_type', 'binary')}]")
                elif isinstance(block.get("text"), str):
                    rendered.append(block["text"])
                else:
                    rendered.append(json.dumps(block, default=str))
        return " ".join(rendered)
    if content is None:
        return ""
    return json.dumps(content, default=str)


def serialize_for_summary(messages: list[Any]) -> str:
    """Lossy flattening used only as summarizer input."""
    lines: list[str] = []
    for message in to_jsonable_python(messages):
        kind = message["kind"]
        for part in message["parts"]:
            part_kind = part["part_kind"]
            if part_kind in {"reasoning", "thinking"}:
                continue
            if kind == "request" and part_kind == "user-prompt":
                lines.append(f"USER:\n{content_text(part)}")
            elif kind == "request" and part_kind == "tool-return":
                lines.append(f"TOOL RESULT:\n{content_text(part)}")
            elif kind == "response" and part_kind == "tool-call":
                args = part.get("args")
                if not isinstance(args, str):
                    args = json.dumps(args, default=str)
                lines.append(f"TOOL CALL: {part.get('tool_name', '')}({args})")
            elif kind == "response" and part_kind in {"text", "final-output"}:
                lines.append(f"ASSISTANT:\n{content_text(part)}")
    return "\n\n".join(lines)


async def summarize_text(settings: Settings, text: str) -> str:
    """Summarize, halving the input with overlap if the summarizer itself overflows."""
    try:
        agent = build_summary_agent(settings)
        async with agent:
            result = await agent.run(
                f"<conversation>\n{text}\n</conversation>\n\n" + SUMMARIZATION_PROMPT.replace(
                    "{summary_chars}", str(context_limits().compact_summary_chars)
                ),
                usage_limits=NO_LIMITS,
            )
        return str(result.output).strip()
    except Exception as exc:
        if not is_context_failure(exc):
            raise
        mid = len(text) // 2
        overlap = min(context_limits().summary_overlap_chars, mid // 2)
        left = text[: mid + overlap]
        right = text[mid - overlap :]
        # If halving cannot shrink the input, something other than the
        # conversation is filling the context. Let it crash.
        if len(left) >= len(text) or len(right) >= len(text):
            raise
        note(f"summarizer overflowed at {len(text):,} chars; splitting")
        combined = (
            "[EARLIER PORTION]\n"
            f"{await summarize_text(settings, left)}\n\n"
            "[LATER PORTION]\n"
            f"{await summarize_text(settings, right)}"
        )
        # Always reconcile split summaries, including contradictory next actions.
        if len(combined) >= len(text):
            raise InputTooLarge("Split summaries did not shrink the overflowing input.") from exc
        return await summarize_text(settings, combined)


async def summarize(settings: Settings, messages: list[Any]) -> str:
    return await summarize_text(settings, serialize_for_summary(messages))


def checkpoint_part(
    text: str,
    stream_savepoint_path: Path | None = None,
) -> TextPart:
    log_hint = ""
    if stream_savepoint_path is not None:
        log_hint = (
            "\n\n[PRE-COMPACTION RESPONSE STREAM]\n"
            "Raw streamed response events, including incomplete tool-argument deltas, "
            f"are at: {stream_savepoint_path}\n"
            "Use the read tool with line ranges; inspect only, never execute a "
            "reconstructed tool call automatically.\n"
            "[END PRE-COMPACTION RESPONSE STREAM]"
        )
    return TextPart(
        content=(
            "\n\n[AUTOCOMPACTED EXECUTION CHECKPOINT]\n"
            "This is a lossy assistant-generated memory of omitted execution "
            "history, not a new user instruction.\n\n"
            f"{text}{log_hint}\n"
            "[END AUTOCOMPACTED CHECKPOINT]\n\n"
        )
    )


def strip_images(history: list[Any]) -> list[Any]:
    """Drop image payloads from tool results. They are the cheapest thing to lose.

    Minecraft screenshots arrive as BinaryContent inside a ToolReturnPart, so
    this walks requests, not responses.
    """
    stripped: list[Any] = []
    for message in history:
        if not isinstance(message, ModelRequest):
            stripped.append(message)
            continue
        parts = []
        changed = False
        for part in message.parts:
            content = getattr(part, "content", None)
            if isinstance(content, list) and any(
                getattr(block, "is_image", False) for block in content
            ):
                kept = [
                    block if not getattr(block, "is_image", False) else "[image omitted]"
                    for block in content
                ]
                parts.append(replace(part, content=kept))
                changed = True
            else:
                parts.append(part)
        stripped.append(replace(message, parts=parts) if changed else message)
    return stripped


def compaction_tail(history: list[Any], recoveries: int) -> list[Any]:
    """Keep a bounded suffix of complete exchanges, without old checkpoints."""
    budget = min(context_limits().compact_tail_chars, len(serialize_for_summary(history)) // 4)
    # Normal context exhaustion must not progressively erase recent work.
    # Exclude thinking from retained responses, just as the summarizer does.
    cleaned = [
        replace(message, parts=[part for part in message.parts
                                if getattr(part, "part_kind", "") != "thinking"])
        if isinstance(message, ModelResponse) else message
        for message in history
    ]
    for index, message in enumerate(cleaned):
        if index == 0 or not isinstance(message, ModelResponse):
            continue
        tail = cleaned[index:]
        text = serialize_for_summary(tail)
        if "[AUTOCOMPACTED EXECUTION CHECKPOINT]" in text:
            continue
        if len(text) <= budget:
            return tail
    return []


class InputTooLarge(RuntimeError):
    """The history contains no generated work to summarize."""


async def compact(
    settings: Settings,
    history: list[Any],
    recoveries: int,
    session: SessionStore | None = None,
) -> list[Any]:
    """Replace execution history with one checkpoint and a bounded recent tail."""
    session = session or active_session
    stream_path = session.save_stream_savepoint() if session is not None else None
    history = strip_images(list(history))
    if not any(isinstance(message, ModelResponse) for message in history):
        raise InputTooLarge("Input way too long, autocompact won't help.")

    # Summarize the whole task, including previous checkpoints and any partial
    # final response, before dropping an unanswered tool call from the tail.
    summary = await summarize(settings, history)
    while len(summary) > context_limits().compact_summary_chars:
        summary = await summarize_text(
            settings,
            "Consolidate this checkpoint to at most "
            f"{context_limits().compact_summary_chars} characters. Preserve requirements and "
            "current task state; replace file contents with paths.\n\n" + summary,
        )
    tail = compaction_tail(drop_unanswered_tail(history), recoveries)
    checkpoint = checkpoint_part(summary, stream_path)
    if tail:
        tail[0] = replace(tail[0], parts=[checkpoint, *tail[0].parts])
    else:
        tail = [ModelResponse(parts=[checkpoint])]
    # Keep the original instructions exact. Everything after them is covered
    # by the consolidated checkpoint, irrespective of synthetic user turns.
    return [history[0], *tail]


# ---------------------------------------------------------------------------
# Sub-agents
#
# The `subagent` tool starts fresh pm-coder sessions and blocks until every
# one finished. Each sub-agent gets the parent's settings and tools, the
# normal system prompt plus an addendum, and no autocompact: one
# agent.run, and when the context is full it stops. What comes back is, per
# sub-agent, how it finished, a summary of its whole chat, and its final
# answer. Submitted sub-agents run concurrently.
# ---------------------------------------------------------------------------

SUBAGENT_ADDENDUM = """
<subagent>
You are a sub-agent. Complete the task in the user prompt.
You cannot ask questions. You cannot compact your context: when your
context is full, you stop. Read only the files you need. Report your
final answer before your context is full. Your final message goes back
to the parent agent, so make it complete and specific: name files,
results, and anything the parent must know.
</subagent>
"""

def make_subagent_tool(
    settings: Settings,
    *,
    bash_machine: Any = None,
    bash_machine_user: str = "user",
) -> Tool[Any]:
    def subagent(
        shared_prompt: str,
        prompts: List[str]
    ) -> str:
        """Start {SUBAGENT_MIN_PROMPTS} to {SUBAGENT_MAX_PROMPTS} sub-agents at once.
        Each sub-agent is a fresh coding agent with the same tools and settings,
        but no memory of this conversation. Give each prompt a complete task
        description: the sub-agent cannot ask questions. Sub-agents cannot
        compact their context. This call blocks until every sub-agent finished.
        For each sub-agent you get: how it finished, a summary of its work, and
        its final answer.

        You can add a shared prompt as text or a file path; it is prepended to
        every prompt. Leave it empty for no shared part. Prompt list items may
        also be text or file paths. When this agent has a BashMachine virtual
        filesystem, file paths are resolved there, never on the host filesystem.

        Use sub-agents to offload independent work and keep your own context
        small. Give them exact goals, scope, expected checks, and relevant
        context. A detailed prompt prevents wasted work. For a very long prompt,
        put the text in a file and pass that path instead.
        """
        try:
            return run_subagents(shared_prompt, prompts)
        except Exception as exc:
            return tool_failure(exc)

    def run_subagents(shared_prompt: str, prompts: List[str]) -> str:
        if len(prompts) < SUBAGENT_MIN_PROMPTS:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        f"give at least {SUBAGENT_MIN_PROMPTS} non-empty prompts, "
                        f"got {len(prompts)}"
                    ),
                }
            )
        if len(prompts) > SUBAGENT_MAX_PROMPTS:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        f"give at most {SUBAGENT_MAX_PROMPTS} non-empty prompts, "
                        f"got {len(prompts)}"
                    ),
                }
            )
        if any(not prompt or not prompt.strip() for prompt in prompts):
            return json.dumps(
                {"success": False, "error": "every sub-agent prompt must be non-empty"}
            )
        note(f"subagents: starting {len(prompts)}")
        # A tool call is capped at five workers, which all run concurrently.
        results: list[dict[str, Any]] = []
        threads: list[threading.Thread] = []
        for prompt in prompts:
            record: dict[str, Any] = {"prompt": prompt}
            results.append(record)
            worker = threading.Thread(
                target=_run_subagent,
                args=(
                    settings,
                    shared_prompt,
                    prompt,
                    record,
                    bash_machine,
                    bash_machine_user,
                ),
            )
            threads.append(worker)
            worker.start()
        for worker in threads:
            worker.join()
        return _render_subagent_results(results)

    assert subagent.__doc__ is not None
    subagent.__doc__ = (
        subagent.__doc__
        .replace("{SUBAGENT_MIN_PROMPTS}", str(SUBAGENT_MIN_PROMPTS))
        .replace("{SUBAGENT_MAX_PROMPTS}", str(SUBAGENT_MAX_PROMPTS))
    )
    return Tool(subagent, takes_ctx=False, name="subagent", sequential=True, strict=False)


def _run_subagent(
    settings: Settings,
    shared_prompt: str,
    prompt: str,
    record: dict[str, Any],
    bash_machine: Any = None,
    bash_machine_user: str = "user",
) -> None:
    if shared_prompt is None:
        shared_prompt = ""
    if prompt is None:
        prompt = ""
    shared_prompt = _subagent_prompt_text(settings, shared_prompt, bash_machine)
    prompt = _subagent_prompt_text(settings, prompt, bash_machine)

    prompt = f"{shared_prompt}\n{prompt}".strip()

    p = prompt.replace("\n", "\\n")
    note(f"subagent start: {p}")
    try:
        record.update(
            asyncio.run(
                _subagent_turn(settings, prompt, bash_machine, bash_machine_user)
            )
        )
    except BaseException as exc:
        record["status"] = f"crashed: {type(exc).__name__}: {exc}"
    note(f"subagent done: {record.get('status', '?')}")


def _subagent_prompt_text(
    settings: Settings, value: str, bash_machine: Any = None
) -> str:
    """Literal prompt text, unless it names a file in the active filesystem."""
    if bash_machine is not None:
        try:
            return bash_machine.read_text(value)
        except Exception:
            # A nonexistent virtual path is ordinary literal prompt text. Do
            # not fall through to the host filesystem from a virtual session.
            return value
    return prompt_text(value, settings.cwd)


async def _subagent_turn(
    settings: Settings,
    prompt: str,
    bash_machine: Any = None,
    bash_machine_user: str = "user",
) -> dict[str, Any]:
    """One sub-agent run: no recovery loop, no compaction, one attempt."""
    discovery = discover_workspace(settings)
    # Its own session dir, so the chat survives for debugging. The HTTP
    # request dumps still go to active_session, which stays the parent's.
    session = SessionStore.open(settings.cwd)
    agent = build_agent(
        settings,
        discovery,
        session,
        bash_machine=bash_machine,
        bash_machine_user=bash_machine_user,
        extra_instructions=SUBAGENT_ADDENDUM,
    )
    async with agent:
        with capture_run_messages() as captured:
            try:
                result = await agent.run(prompt, usage_limits=NO_LIMITS)
            except Exception as exc:
                # Out of memory or a broken response: the sub-agent stops.
                # The chat is still worth a summary for the parent.
                messages = list(captured)
                summary = await _subagent_summary(settings, session, messages)
                return {
                    "status": f"stopped: {type(exc).__name__}: {exc}",
                    "summary": summary,
                }
    session.save_messages(result.all_messages())
    return {
        "status": "completed",
        "summary": await _subagent_summary(settings, session, result.all_messages()),
        "output": str(result.output),
    }


async def _subagent_summary(
    settings: Settings, session: SessionStore, messages: list[Any]
) -> str:
    """Short summary of a sub-agent's thinking and tool calls."""
    if not messages:
        return "(the sub-agent made no calls)"
    try:
        return await summarize(settings, messages)
    except Exception:
        # The endpoint is down or the chat cannot be summarized. The raw
        # call list is still a summary.
        return session.tool_stats_report()


def _render_subagent_results(results: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for number, record in enumerate(results):
        lines = [f"=== sub-agent {number}: {record.get('status', '?')} ==="]
        # lines.append(f"prompt: {record['prompt'][:200]}")
        if "output" in record:
            lines.append(f"Final answer:\n{record['output']}")
        else:
            lines.append(f"Summary of unfinished/failed execution:\n{record.get('summary', '(none)')}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# The loop iteration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnResult:
    """One completed turn. Also the JSON payload auto mode prints."""

    response: str
    run_id: str
    duration_seconds: float
    tokens_used: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resume_prompt(history: list[Any], fallback: str) -> str | None:
    """What to send next, given a history we are about to retry.

    ``None`` tells Pydantic AI to re-issue the trailing request as-is: the
    model continues from the tool results already in the history instead of
    being handed a new user turn it would answer from scratch. Only when the
    history ends on a response -- nothing left to answer -- does the fallback
    text get sent as a real prompt.
    """
    if history and isinstance(history[-1], ModelRequest):
        return None
    return fallback

async def run_turn(
    agent: Agent[Any, str],
    settings: Settings,
    session: SessionStore,
    prompt: str,
) -> TurnResult:
    """Run one prompt to completion. The only place a model turn happens.

    Interactive mode calls this once per line the user types; auto mode calls
    it once and exits. It has no failure it will not absorb:

    * out of context -> compact the history and resume from the last tool
      result;
    * the model stuck on its own output (unparseable generation, or arguments
      it cannot repair) -> nudge it with a fresh user turn;
    * anything else -> say so, wait, and retry the same turn.

    The agent must already be entered (``async with agent``) so MCP servers
    stay connected across retries instead of reconnecting on every hiccup.
    """
    history = session.load_messages()
    next_prompt: str | None = prompt
    recoveries = 0
    started = time.perf_counter()
    active_session.turn_id = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))
    active_session.auto_compact_cnt = 0

    async def recover(reason: str) -> str | None:
        nonlocal history, recoveries
        note(f"{reason}; compacting (recovery level {recoveries})")
        history = await compact(settings, history, recoveries, session)
        recoveries += 1
        session.save_messages(history)
        # The stats view rides into the next request as a fake user turn:
        # what was done since the last checkpoint, so the model does not
        # redo it, and a read loop is visible to the model itself.
        session.pending_alert = (
            (session.pending_alert + "\n\n" if session.pending_alert else "")
            + "[checkpoint stats]\n"
            + session.tool_stats_report()
        )
        session.tool_calls.clear()
        active_session.auto_compact_cnt += 1
        note("compacted to one checkpoint plus recent exchanges; resuming")
        return resume_prompt(history, CONTEXT_RECOVERY_PROMPT)

    while True:
        try:
            with capture_run_messages() as captured:
                session.live_history = captured
                with Agent.parallel_tool_call_execution_mode("sequential"):
                    result = await agent.run(
                        next_prompt,
                        message_history=history or None,
                        usage_limits=NO_LIMITS,
                    )
        except Exception as exc:
            # `captured` holds the turn so far, including tool calls that
            # already ran. Keeping it means a retry resumes instead of
            # repeating side effects; Pydantic AI closes any dangling call.
            history = list(captured) or history
            session.save_messages(history)

            if is_unprocessed_tool_calls_error(exc):
                # The endpoint refused the prompt because a tool call sits
                # unanswered in the history. Cut that call off and retry.
                history = drop_last_tool_call(history)
                session.save_messages(history)
                next_prompt = resume_prompt(history, prompt)
                continue

            if is_model_intelligence_failure(exc):
                # The model is wedged on output it cannot fix: a generation
                # the endpoint could not parse, or tool arguments it emitted
                # byte-identical despite the validation error being fed
                # back. Retrying the same history resamples the same
                # distribution, so inject a user turn to move it. The
                # failure can leave the last call without a result -- retry
                # exhaustion raises before one is appended -- and no new
                # prompt may follow such a call, so trim it. The error
                # feedback from the attempts that did land stays visible.
                history = drop_unanswered_tail(history)
                session.save_messages(history)
                next_prompt = FAKE_USER_RESUME
            elif is_context_failure(exc) or hit_generation_limit(history):
                # compact
                next_prompt = await recover(
                    f"out of room ({type(exc).__name__})",
                )
            elif is_unsupported_input(exc):
                raise Exception(f"{type(exc).__name__}: endpoint has no image/mtmd support!")
            else:
                # reconnect. The exception may have struck between the model
                # emitting a tool call and that call's result landing;
                # resume_prompt would then answer the dangling call with a
                # fresh prompt, which pydantic-ai rejects -- and keeps
                # rejecting, because every retry rebuilds the same history.
                # Trim the call so the resume continues from the last real
                # result instead of tripping over the stub.
                history = drop_unanswered_tail(history)
                session.save_messages(history)
                note(f"{type(exc).__name__}: {exc}, trying reconnect...")
                next_prompt = resume_prompt(history, prompt)
                await asyncio.sleep(RETRY_DELAY_SECONDS)
            continue

        history = result.all_messages()
        session.save_messages(history)

        if hit_generation_limit(history):
            # The response came back whole enough to parse but stopped
            # mid-thought. Same cause as the raising cases, same cure.
            next_prompt = await recover(
                "response hit the generation limit"
            )
            continue

        turn = TurnResult(
            response=str(result.output),
            run_id=session.run_id,
            duration_seconds=round(time.perf_counter() - started, 3),
            tokens_used=to_jsonable_python(asdict(result.usage)),
        )
        session.append_run({"timestamp": utc_now(), "prompt": prompt, **turn.as_dict()})
        return turn


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def prompt_text(prompt_or_path: str | Path, cwd: Path) -> str:
    """Literal text, unless it names a readable file."""
    candidate = Path(prompt_or_path).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return str(prompt_or_path)


def print_startup(
    settings: Settings, discovery: DiscoveryResult, session: SessionStore
) -> None:
    selected = discovery.selected_skill.name if discovery.selected_skill else "(none)"
    lines = [
        f"\n{APP_NAME}",
        f"  cwd:            {settings.cwd}",
        f"  endpoint:       {settings.base_url}",
        f"  model:          {settings.model}",
        f"  shell:          {settings.shell_kind} ({settings.shell_executable})",
        f"  mcp servers:    {', '.join(discovery.mcp_server_names) or 'none configured'}",
        f"  skills:         {len(discovery.skills)}",
        f"  selected skill: {selected}",
        f"  instructions:   {len(discovery.instruction_files)} file(s)",
        f"  context window: {settings.context_window:,}",
        f"  verbose:        {'on' if settings.verbose else 'off'}",
        f"  run id:         {session.run_id}",
        f"  session dir:    {session.path}",
    ]
    lines += [f"  skill warning:  {error}" for error in discovery.skill_errors]
    print("\n".join(lines), file=sys.stderr, flush=True)


@asynccontextmanager
async def open_session(
    settings: Settings,
    *,
    run_id: str | None = None,
    log_root: str | Path = DEFAULT_LOG_ROOT,
) -> Any:
    """Yield a live (agent, discovery, store) for as many turns as you want.

    The agent is entered once so MCP servers stay connected for the whole
    session; :func:`run_turn` retries inside that, never around it.
    """
    global active_session
    discovery = discover_workspace(settings)
    store = SessionStore.open(settings.cwd, run_id, log_root=Path(log_root).expanduser())
    active_session = store
    active_session.context_window = settings.context_window
    agent = build_agent(
        settings, discovery, store, with_subagent_tool=True, capture_stream=True
    )
    print_startup(settings, discovery, store)
    async with agent:
        yield agent, discovery, store


@asynccontextmanager
async def open_bash_machine_session(
    settings: Settings,
    bash_machine: Any,
    *,
    run_id: str | None = None,
    log_root: str | Path = DEFAULT_LOG_ROOT,
    user: str = "user",
) -> Any:
    """Yield a live (agent, discovery, store) backed by an in-memory BashMachine.

    Like :func:`open_session`, but the shell tool routes into ``bash_machine``
    instead of spawning a real subprocess. The same BashMachine can be shared
    across multiple sessions in different threads.

    ``user`` is the virtual user name inside the BashMachine.
    """
    global active_session
    discovery = discover_workspace(settings)
    store = SessionStore.open(settings.cwd, run_id, log_root=Path(log_root).expanduser())
    active_session = store
    active_session.context_window = settings.context_window
    agent = build_agent(
        settings, discovery, store,
        bash_machine=bash_machine,
        bash_machine_user=user,
        with_subagent_tool=True,
        capture_stream=True,
    )
    print_startup(settings, discovery, store)
    note(f"bash-machine: user={user!r}")
    async with agent:
        yield agent, discovery, store


async def async_run_auto(
    prompt_or_path: str | Path,
    *,
    run_id: str | None = None,
    log_root: str | Path = DEFAULT_LOG_ROOT,
    **settings_kwargs: Any,
) -> TurnResult:
    """Run one prompt to completion and return its result.

    ``prompt_or_path`` is a path to a UTF-8 text file when it names one, and
    literal prompt text otherwise. Pass the ``run_id`` from an earlier result
    to continue that exact conversation; ``messages.json`` is updated before
    this returns. Remaining keywords go to :func:`build_settings`.
    """
    settings = build_settings(**settings_kwargs)
    async with open_session(settings, run_id=run_id, log_root=log_root) as (
        agent,
        _discovery,
        store,
    ):
        return await run_turn(
            agent, settings, store, prompt_text(prompt_or_path, settings.cwd)
        )


def run_auto(prompt_or_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Synchronous :func:`async_run_auto`, for ``run_auto(input())`` style loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(async_run_auto(prompt_or_path, **kwargs)).as_dict()
    raise RuntimeError("run_auto cannot run inside an event loop; use async_run_auto")


async def async_run_auto_with_bash_machine(
    prompt_or_path: str | Path,
    bash_machine: Any,
    *,
    user: str = "user",
    run_id: str | None = None,
    log_root: str | Path = DEFAULT_LOG_ROOT,
    **settings_kwargs: Any,
) -> TurnResult:
    """Like :func:`async_run_auto`, but the shell tool runs in ``bash_machine``.

    ``bash_machine`` is a :class:`BashMachine` instance. It can be shared
    across threads.
    """
    settings = build_settings(**settings_kwargs)
    async with open_bash_machine_session(
        settings, bash_machine, run_id=run_id, log_root=log_root, user=user,
    ) as (agent, _discovery, store):
        return await run_turn(
            agent, settings, store, prompt_text(prompt_or_path, settings.cwd)
        )


def run_auto_with_bash_machine(
    prompt_or_path: str | Path,
    bash_machine: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Synchronous :func:`async_run_auto_with_bash_machine`."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            async_run_auto_with_bash_machine(prompt_or_path, bash_machine, **kwargs)
        ).as_dict()
    raise RuntimeError(
        "run_auto_with_bash_machine cannot run inside an event loop; "
        "use async_run_auto_with_bash_machine"
    )


def read_user_prompt() -> str | None:
    try:
        first_line = input("You> ")
    except EOFError:
        return None
    if first_line.strip().casefold() != "/paste":
        return first_line
    print("Paste mode. Enter /end on its own line to submit.")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().casefold() == "/end":
            break
        lines.append(line)
    return "\n".join(lines)


async def interactive_loop(
    agent: Agent[Any, str],
    settings: Settings,
    discovery: DiscoveryResult,
    store: SessionStore,
) -> None:
    while True:
        prompt = read_user_prompt()
        if prompt is None:
            return
        command = prompt.strip().casefold()
        if command in {"/quit", "/exit", "quit", "exit"}:
            return
        if command == "/clear":
            store.clear()
            continue
        if command == "/info":
            print_startup(settings, discovery, store)
            continue
        if not prompt.strip():
            continue
        turn = await run_turn(agent, settings, store, prompt)
        print(f"\nAgent> {turn.response}\n", flush=True)


async def async_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = settings_from_args(args)
    prompt = args.prompt_file or args.prompt
    if args.mode == "auto":
        if prompt is None:
            raise ValueError("auto mode requires prompt text or --prompt-file")
        async with open_session(settings, run_id=args.run_id, log_root=args.log_root) as (
            agent,
            _discovery,
            store,
        ):
            turn = await run_turn(
                agent, settings, store, prompt_text(prompt, settings.cwd)
            )
        print(json.dumps(turn.as_dict(), ensure_ascii=False), flush=True)
        return
    if prompt is not None:
        raise ValueError("a prompt requires --mode auto")
    async with open_session(settings, run_id=args.run_id, log_root=args.log_root) as (
        agent,
        discovery,
        store,
    ):
        await interactive_loop(agent, settings, discovery, store)


def main() -> None:
    # Windows picks the console codepage for stdout, so a single em-dash in a
    # model response lands as cp1252 byte 0x97 and the auto-mode JSON stops
    # being valid UTF-8 for whatever is parsing it. stderr gets the same
    # treatment so a stray character in tool output cannot raise mid-run.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        note("interrupted")
        os._exit(1)
    except InputTooLarge as exc:
        note(str(exc))
        os._exit(1)


if __name__ == "__main__":
    main()
