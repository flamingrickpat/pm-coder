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
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
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
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 8_192

# Seconds to wait before retrying after any failure that is not a context
# problem. One fixed delay: a week-long run has no deadline to race.
RETRY_DELAY_SECONDS = 30.0

# 0 means "ask the endpoint": use its advertised n_ctx minus this margin so
# generation stays under the serving cap. Used only to size compaction.
DEFAULT_CONTEXT_WINDOW = 0
CONTEXT_SAFETY_MARGIN = 2_048
FALLBACK_CONTEXT_WINDOW = 65_536

# Fraction of the pre-compaction history a compaction keeps verbatim, split
# across its two edges. A first compaction should land at roughly half the
# size it started from (90k -> ~45k), not at a fixed few thousand tokens
# that vaporize an all-night session's detail. Consecutive recoveries still
# halve whatever was chosen, so a history that keeps overflowing keeps
# converging instead of stalling.
COMPACT_KEEP_FRACTION = 0.5
# Per-edge cap relative to the context window: the verbatim edges plus the
# checkpoint summary must leave room for the next generation to run.
EDGE_WINDOW_FRACTION = 0.25
MIN_EDGE_TOKENS = 256
# Growth since the previous compaction that counts as the model having done
# real work, rather than having overflowed again straight away.
PROGRESS_TOKENS = 2_048
# A single turn larger than this fraction of the window is compacted on its
# own before anything else: one fat coding turn must not permanently occupy
# every future request.
FAT_TURN_FRACTION = 0.20
SUMMARY_MAX_TOKENS = 8_192
# The summarizer is fed text, not tokens; splitting by characters is enough.
SUMMARY_OVERLAP_CHARS = 6_000
# Flat per-image budget for the context estimator. The real vision cost is
# deployment specific, but compaction only needs relative sizes.
IMAGE_TOKENS = 400

# One agent.run can last all night, so the history is snapshotted mid-turn
# after a tool call, at most this often.
SNAPSHOT_SECONDS = 30.0
# Fraction of the context window above which a truncated response is treated
# as the context running out. Below it, the response merely outgrew
# --max-tokens, and summarizing would throw away detail for nothing.
COMPACT_ABOVE = 0.75

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


def _omit_older_images(messages: list[ModelMessage], keep: int) -> list[ModelMessage]:
    """Swap every image but the newest ``keep`` for an OMITTED placeholder."""

    kept = 0

    # Walk content newest -> oldest.
    def prune(x):
        nonlocal kept

        if _is_image(x):
            if kept < keep:
                kept += 1
                return x
            return OMITTED

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

class SessionStore:
    """One conversation on disk as Pydantic AI model messages.

    ``messages.json`` is replayed verbatim into ``message_history``, so
    resuming a session does not re-summarize or re-prompt anything.
    """

    schema = "pm-coder-session.v1"

    def __init__(self, path: Path, cwd: Path) -> None:
        self.path = path
        self.cwd = cwd.resolve()
        self.metadata_path = path / "session.json"
        self.messages_path = path / "messages.json"
        self.runs_path = path / "runs.jsonl"
        self.path.mkdir(parents=True, exist_ok=True)
        # Set by run_turn to the list Pydantic AI appends to as a turn runs,
        # so a mid-turn snapshot writes the whole conversation and not just
        # the fragment produced so far.
        self.live_history: list[Any] | None = None
        self.last_snapshot = 0.0

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
            sequence = next(self._log_counter)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            stem = active_session.path / f"{timestamp}_{sequence:06d}"
            stem.with_suffix(".compact.json").write_bytes(raw)
            stem.with_suffix(".pretty.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
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
    temperature: float
    max_tokens: int | None
    disable_thinking: bool
    skill: str | None
    verbose: bool
    context_window: int = Field(gt=0)


def probe_endpoint(
    base_url: str, api_key: str, *, timeout: float = 10.0
) -> dict[str, Any] | None:
    """Return the first model the endpoint advertises, or None if unreachable.

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
    for entry in payload.get("data") or []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        meta = entry.get("meta")
        n_ctx = meta.get("n_ctx") if isinstance(meta, dict) else None
        return {
            "id": str(entry["id"]),
            "n_ctx": n_ctx if isinstance(n_ctx, int) and n_ctx > 0 else None,
        }
    return None


def wait_for_endpoint(base_url: str, api_key: str) -> dict[str, Any]:
    """Block until the endpoint answers. Startup must survive a cold server."""
    while True:
        capabilities = probe_endpoint(base_url, api_key)
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
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
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
        capabilities = wait_for_endpoint(resolved_base_url, resolved_api_key)
    if resolved_model is None:
        resolved_model = capabilities["id"]
    if context_window <= 0:
        served = capabilities["n_ctx"]
        context_window = (
            max(served - CONTEXT_SAFETY_MARGIN, CONTEXT_SAFETY_MARGIN)
            if served
            else FALLBACK_CONTEXT_WINDOW
        )

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
        temperature=temperature,
        max_tokens=max_tokens,
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
            "endpoint's advertised n_ctx and subtracts a safety margin."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("LOCAL_AGENT_TEMPERATURE", DEFAULT_TEMPERATURE)),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=env_int("LOCAL_AGENT_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        help="Generated tokens per response. Pass 0 to let the server decide.",
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
        temperature=args.temperature,
        max_tokens=args.max_tokens or None,
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


def make_shell_tool(settings: Settings) -> Tool[Any]:
    backend = shell_backend(settings)

    def host_shell(command: str, timeout_seconds: int = settings.shell_timeout) -> str:
        """Execute a host-shell script in the selected agent workspace."""
        print(f"\n[{backend.kind}]\n{command.rstrip()}", file=sys.stderr, flush=True)
        timeout = None if timeout_seconds <= 0 else timeout_seconds
        script_path: str | None = None
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
            completed = subprocess.run(
                backend.invocation(script_path),
                cwd=settings.cwd,
                env=os.environ.copy(),
                capture_output=True,
                timeout=timeout,
                check=False,
            )
            stdout = completed.stdout.decode("utf-8", errors="replace")
            stderr = completed.stderr.decode("utf-8", errors="replace")
            print(f"[{backend.kind} exit {completed.returncode}]", file=sys.stderr, flush=True)
            return (
                f"exit_code: {completed.returncode}\n"
                f"stdout:\n{stdout or '(empty)'}\n"
                f"stderr:\n{stderr or '(empty)'}"
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
            print(f"[{backend.kind} timed out]", file=sys.stderr, flush=True)
            return (
                "timed_out: true\n"
                f"timeout_seconds: {timeout_seconds}\n"
                f"stdout_before_timeout:\n{stdout or '(empty)'}\n"
                f"stderr_before_timeout:\n{stderr or '(empty)'}"
            )
        finally:
            if script_path is not None:
                with suppress(OSError):
                    os.remove(script_path)

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
# read_image is the visual variant: Pydantic AI turns a BinaryContent tool
# return into a base64 image_url user message, so the tool just hands the
# bytes back and the framework does the rest.
# ---------------------------------------------------------------------------


def resolve_path(settings: Settings, path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = settings.cwd / candidate
    return candidate


def read_file_text(path: Path) -> str:
    """Decode a file without universal-newline translation.

    ``Path.read_text`` rewrites CRLF to LF in memory, which would hide a
    file's real line endings from :func:`newline_style` and silently convert
    every edited file to LF.
    """
    return path.read_bytes().decode("utf-8", errors="replace")


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


IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _edit_text(
    text: str, old_string: str, new_string: str, replace_all: bool
) -> tuple[str | None, str]:
    """Apply an exact string replacement in memory.

    Returns (updated_text, message). updated_text is None on failure.
    """
    old = old_string.replace("\r\n", "\n")
    new = new_string.replace("\r\n", "\n")
    if not old:
        return None, "error: old_string is empty; use write to create a file"
    count = text.count(old)
    if count == 0:
        return (
            None,
            "error: no match for old_string. It must match the "
            "file exactly, including whitespace and indentation."
            + match_hint(text, old),
        )
    if count > 1 and not replace_all:
        return (
            None,
            f"error: found {count} matches for old_string. Add "
            "surrounding context to make it unique, or pass replace_all=true.",
        )
    line = text[: text.index(old)].count("\n") + 1
    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    where = f"{count} occurrences" if replace_all else f"line {line}"
    return updated, f"edited ({where}, file is now {len(split_lines(updated))} lines)"


def make_file_tools(settings: Settings) -> list[Tool[Any]]:
    def read(path: str, offset: int = 1, limit: int = 0) -> str:
        """Read a text file. Line numbers are prepended; they are for reference
        only, since `edit` addresses content by exact text. `offset` is the
        1-indexed first line and `limit` the maximum number of lines (0 = all).
        """
        target = resolve_path(settings, path)
        print(f"\n[read {target}]", file=sys.stderr, flush=True)
        if not target.is_file():
            return f"error: file does not exist: {target}"
        lines = split_lines(read_file_text(target))
        start = max(1, offset)
        end = len(lines) if limit <= 0 else min(len(lines), start + limit - 1)
        header = f"{target} ({len(lines)} lines)"
        if start > 1 or end < len(lines):
            header += f", showing lines {start}-{end}"
        body = "\n".join(f"{number}: {lines[number - 1]}" for number in range(start, end + 1))
        return f"{header}\n{body}"

    def read_image(path: str):
        """Attach a JPG or PNG image to the conversation so the model can see
        it. The image is returned alongside a short text confirmation; the
        model receives both as visual input. Use this for screenshots,
        diagrams, or any picture the task refers to.
        """
        target = resolve_path(settings, path)
        print(f"\n[read_image {target}]", file=sys.stderr, flush=True)
        if not target.is_file():
            return [f"error: file does not exist: {target}"]
        media_type = IMAGE_MEDIA_TYPES.get(target.suffix.lower())
        if media_type is None:
            return [
                f"error: unsupported image type {target.suffix or '(none)'} in "
                f"{target}; use .jpg, .jpeg, or .png"
            ]
        data = target.read_bytes()
        if not data:
            return [f"error: file is empty: {target}"]
        return [
            BinaryContent(data, media_type=media_type),
            f"image attached: {target} ({len(data):,} bytes)",
        ]

    def write(path: str, content: str) -> str:
        """Create a new file, or completely rewrite an existing one, as UTF-8.

        Use this for a new file or a wholesale rewrite. To change part of an
        existing file use `edit` instead: rewriting a long file spends the
        whole response budget and risks being cut off mid-call.
        """
        target = resolve_path(settings, path)
        existed = target.is_file()
        body = content.replace("\r\n", "\n")
        if existed:
            body = body.replace("\n", newline_style(read_file_text(target)))
        atomic_write_bytes(target, body.encode("utf-8"))
        verb = "overwrote" if existed else "created"
        print(f"\n[write {verb} {target}]", file=sys.stderr, flush=True)
        return f"{verb} {target} ({len(split_lines(body))} lines)"

    def edit(
        path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        """Replace an exact string in a file.

        `old_string` must match the file exactly, including whitespace and
        indentation -- copy it verbatim from a `read`. It must match exactly
        once unless `replace_all` is set. Prefer one larger replacement of a
        coherent block over several small interleaved ones.
        """
        target = resolve_path(settings, path)
        print(f"\n[edit {target}]", file=sys.stderr, flush=True)
        if not target.is_file():
            return f"error: file does not exist: {target}"
        raw = read_file_text(target)
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
        atomic_write_bytes(target, updated.replace("\n", newline).encode("utf-8"))
        where = f"{count} occurrences" if replace_all else f"line {line}"
        return f"edited {target} ({where}, file is now {len(split_lines(updated))} lines)"

    return [
        Tool(read, takes_ctx=False, name="read", sequential=True, strict=False),
        Tool(read_image, takes_ctx=False, name="read_image", sequential=True, strict=False),
        Tool(write, takes_ctx=False, name="write", sequential=True, strict=False),
        Tool(edit, takes_ctx=False, name="edit", sequential=True, strict=False),
    ]


def make_virtual_file_tools(bash_machine: Any) -> list[Tool[Any]]:
    """File tools that read and write in an in-memory BashMachine.

    Paths are virtual, e.g. ``/home/user/notes.txt`` or relative paths.
    Relative paths resolve against the admin shell's cwd (``/home/user``).
    """

    def read(path: str, offset: int = 1, limit: int = 0) -> str:
        print(f"\n[read :{path}]", file=sys.stderr, flush=True)
        try:
            raw = bash_machine.read_text(path)
        except Exception as exc:
            return f"error: cannot read {path}: {exc}"
        lines = split_lines(raw)
        start = max(1, offset)
        end = len(lines) if limit <= 0 else min(len(lines), start + limit - 1)
        header = f"{path} ({len(lines)} lines)"
        if start > 1 or end < len(lines):
            header += f", showing lines {start}-{end}"
        body = "\n".join(
            f"{number}: {lines[number - 1]}" for number in range(start, end + 1)
        )
        return f"{header}\n{body}"

    def read_image(path: str):
        print(f"\n[read_image :{path}]", file=sys.stderr, flush=True)
        try:
            data = bash_machine.read_binary(path)
        except Exception as exc:
            return [f"error: cannot read {path}: {exc}"]
        suffix = Path(path).suffix.lower()
        media_type = IMAGE_MEDIA_TYPES.get(suffix)
        if media_type is None:
            return [
                f"error: unsupported image type {suffix or '(none)'} in "
                f"{path}; use .jpg, .jpeg, or .png"
            ]
        if not data:
            return [f"error: file is empty: {path}"]
        return [
            BinaryContent(data, media_type=media_type),
            f"image attached: {path} ({len(data):,} bytes)",
        ]

    def write(path: str, content: str) -> str:
        body = content.replace("\r\n", "\n")
        try:
            bash_machine.write_text(path, body)
        except Exception as exc:
            return f"error: cannot write {path}: {exc}"
        print(f"\n[write {path}]", file=sys.stderr, flush=True)
        return f"created/overwrote {path} ({len(split_lines(body))} lines)"

    def edit(
        path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        print(f"\n[edit :{path}]", file=sys.stderr, flush=True)
        try:
            raw = bash_machine.read_text(path)
        except Exception as exc:
            return f"error: cannot read {path}: {exc}"
        text = raw.replace("\r\n", "\n")
        updated, message = _edit_text(text, old_string, new_string, replace_all)
        if updated is None:
            return message
        try:
            bash_machine.write_text(path, updated)
        except Exception as exc:
            return f"error: cannot write {path}: {exc}"
        return f"edited {path} ({message})"

    return [
        Tool(read, takes_ctx=False, name="read", sequential=True, strict=False),
        Tool(read_image, takes_ctx=False, name="read_image", sequential=True, strict=False),
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
        You are a local coding agent operating directly in one agent workspace.

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

        Make one tool call per response and wait for its result before
        deciding the next one. Tool calls run strictly in order anyway, so
        batching several into one response buys nothing and costs you the
        chance to react to what each one returned.

        Use `read`, `write`, and `edit` for files. `edit` replaces an exact,
        unique string: copy `old_string` verbatim out of a `read`, including
        its indentation. Use `write` for a new file or a deliberate full
        rewrite. Prefer one larger `edit` of a coherent block over several
        small interleaved ones. Use `read_image` to look at a JPG or PNG
        image; the picture itself becomes part of the conversation.

        `{kind}` starts in the selected workspace. Use it for everything
        else: running commands, searching, and verifying results. Project
        instructions define the exact writable paths. Do not escape the
        selected workspace.

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


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def make_model(settings: Settings, label: str) -> Any:
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
        return result


def build_agent(
    settings: Settings,
    discovery: DiscoveryResult,
    session: SessionStore,
    *,
    bash_machine: Any = None,
    bash_machine_user: str = "user",
) -> Agent[Any, str]:
    shell_tool = (
        make_bash_machine_tool(bash_machine, bash_machine_user)
        if bash_machine is not None
        else make_shell_tool(settings)
    )
    file_tools = (
        make_virtual_file_tools(bash_machine)
        if bash_machine is not None
        else make_file_tools(settings)
    )
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
    )
    mcp_tools = (
        load_mcp_toolsets(settings.mcp_config) if settings.mcp_config is not None else []
    )
    toolset = SnapshotToolset(
        CombinedToolset([own_tools, *mcp_tools]),
        session=session,
    )
    return Agent(
        model=make_model(settings, "agent"),
        instructions=system_prompt,
        toolsets=[toolset],
        model_settings=OpenAIChatModelSettings(
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            parallel_tool_calls=False,
            extra_body=thinking_body(settings),
        ),
        capabilities=[
            ProcessHistory(keep_recent_images),
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
            max_tokens=SUMMARY_MAX_TOKENS,
            parallel_tool_calls=False,
            extra_body=thinking_body(settings),
        ),
        retries=1,
    )


# ---------------------------------------------------------------------------
# Context estimation
#
# Only relative sizes matter here: which turn is the fattest, and where the
# verbatim edges of a compaction should fall. Four characters per token is
# close enough for that, and images get one flat budget rather than a probe.
# ---------------------------------------------------------------------------


def _json_length(value: Any) -> int:
    return len(json.dumps(value, default=str, sort_keys=True))


def _block_chars(block: Any) -> int:
    """Characters attributable to one element of a multi-part content list."""
    if isinstance(block, str):
        return len(block)
    if not isinstance(block, dict):
        return _json_length(block)
    if block.get("kind") == "binary":
        # An image costs vision tokens, not the length of its base64 payload.
        return IMAGE_TOKENS * 4
    if isinstance(block.get("text"), str):
        return len(block["text"])
    if block.get("type") == "tool-call" or block.get("part_kind") == "tool-call":
        return len(block.get("tool_name") or "") + _json_length(block.get("args"))
    return _json_length(block)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    chars = 0
    for part in message["parts"]:
        kind = part.get("part_kind") or part.get("kind")
        if kind == "tool-call":
            chars += len(part.get("tool_name") or "") + _json_length(part.get("args"))
            continue
        content = part.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            chars += sum(_block_chars(block) for block in content)
        elif content is not None:
            chars += _json_length(content)
    return max(1, math.ceil(chars / 4))


def estimate_context_tokens(data: list[dict[str, Any]]) -> int:
    """Prefer the endpoint's own count from the newest response it answered."""
    for index in range(len(data) - 1, -1, -1):
        message = data[index]
        if message.get("kind") != "response":
            continue
        usage = message.get("usage")
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        if isinstance(total, int) and total > 0:
            return total + sum(estimate_message_tokens(m) for m in data[index + 1 :])
        break
    return sum(estimate_message_tokens(m) for m in data)


def count_tokens(messages: list[Any]) -> int:
    return estimate_context_tokens(to_jsonable_python(messages))


# ---------------------------------------------------------------------------
# Compaction
#
# Reactive only: nothing is summarized until the endpoint says the context is
# full or a response comes back truncated. Three strategies escalate, and each
# consecutive recovery halves the verbatim edges it preserves, so a history
# that keeps overflowing keeps shrinking instead of stalling.
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
memory. Prefer saying what should be re-read/re-checked over reproducing it.

Distinguish facts from unresolved hypotheses when that matters.
Do not invent anything.

Write the checkpoint as compact Markdown.
"""

TRUNCATION_FEEDBACK = (
    "Your previous response was cut off when it reached the per-response token "
    "limit. Whatever it managed to produce is above, unfinished. Continue from "
    "exactly where it stopped instead of starting over, and keep each single "
    "response smaller: write a long file as several `edit` calls rather than "
    "one enormous `write`."
)

FAKE_USER_RESUME = "/resume"

CONTEXT_RECOVERY_PROMPT = (
    "The earlier conversation reached the model context window and was "
    "summarized into the checkpoint above. Continue the CURRENT task from that "
    "checkpoint. Do NOT restart from scratch. Keep making concrete progress on "
    "the original goal, verify with fresh state, and return the required "
    "result when you are done."
)

CONTEXT_MARKERS = (
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
    "n_ctx",
)

# Errors that mean the endpoint rejected what the model *produced*, not what we
# sent. llama.cpp answers 500 when the tool call the model emitted is not
# parsable JSON; that happens at any context size, while a genuine out-of-room
# rejection is a 4xx carrying one of the markers above. Reading it as "no room"
# cost a session its history once: a 4,798-token conversation was compacted to
# 545 tokens, because the forced path skips the size check.
GENERATION_FAILURE_MARKERS = ("failed to parse", "as json",)


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
                f"<conversation>\n{text}\n</conversation>\n\n{SUMMARIZATION_PROMPT}",
                usage_limits=NO_LIMITS,
            )
        return str(result.output).strip()
    except Exception as exc:
        if not is_context_failure(exc):
            raise
        mid = len(text) // 2
        overlap = min(SUMMARY_OVERLAP_CHARS, mid // 2)
        left = text[: mid + overlap]
        right = text[mid - overlap :]
        # If halving cannot shrink the input, something other than the
        # conversation is filling the context. Let it crash.
        if len(left) >= len(text) or len(right) >= len(text):
            raise
        note(f"summarizer overflowed at {len(text):,} chars; splitting")
        return (
            "[EARLIER PORTION]\n"
            f"{await summarize_text(settings, left)}\n\n"
            "[LATER PORTION]\n"
            f"{await summarize_text(settings, right)}"
        )


async def summarize(settings: Settings, messages: list[Any]) -> str:
    return await summarize_text(settings, serialize_for_summary(messages))


def checkpoint_part(text: str) -> TextPart:
    return TextPart(
        content=(
            "\n\n[AUTOCOMPACTED EXECUTION CHECKPOINT]\n"
            "This is a lossy assistant-generated memory of omitted execution "
            "history, not a new user instruction.\n\n"
            f"{text}\n"
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


def is_user_turn(message: Any) -> bool:
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart) for part in message.parts
    )


def turn_ranges(history: list[Any]) -> list[tuple[int, int]]:
    """Half-open [start, end) index ranges, one per user-initiated turn."""
    starts = [index for index, message in enumerate(history) if is_user_turn(message)]
    return [
        (start, starts[n + 1] if n + 1 < len(starts) else len(history))
        for n, start in enumerate(starts)
    ]


async def compact_one_turn(
    settings: Settings, turn: list[Any], edge_tokens: int
) -> list[Any] | None:
    """Summarize the interior of one turn, keeping verbatim head and tail.

    A turn is UserRequest, Response, ToolReturn, Response, ToolReturn, ... so
    cutting immediately before a ModelResponse leaves the head ending in a
    request and the tail starting in a response. The checkpoint is prepended
    to that response, which keeps every tool-call/tool-return pair intact.
    """
    cuts = [index for index, message in enumerate(turn) if isinstance(message, ModelResponse)]
    # A turn smaller than both edges cannot keep them verbatim; quarter the
    # turn instead so this still lands near half its size rather than
    # falling through to collapse_largest_turn's request-plus-summary.
    edge_tokens = min(edge_tokens, max(MIN_EDGE_TOKENS, count_tokens(turn) // 4))
    head_cuts = [index for index in cuts if count_tokens(turn[:index]) >= edge_tokens]
    tail_cuts = [index for index in cuts if count_tokens(turn[index:]) >= edge_tokens]
    if not head_cuts or not tail_cuts:
        return None
    head_cut, tail_cut = head_cuts[0], tail_cuts[-1]
    if tail_cut <= head_cut:
        return None
    # Summarize the WHOLE turn, not merely the omitted middle: the preserved
    # edges give exact detail, and this gives coherent state around them.
    summary = await summarize(settings, turn)
    head = list(turn[:head_cut])
    tail = list(turn[tail_cut:])
    tail[0] = replace(tail[0], parts=[checkpoint_part(summary), *tail[0].parts])
    return head + tail


async def compact_middle_turns(
    settings: Settings, history: list[Any], edge_tokens: int
) -> list[Any] | None:
    """Keep whole turns at both ends verbatim; summarize the whole turns between."""
    ranges = turn_ranges(history)
    if len(ranges) < 3:
        return None
    left = 0
    kept = 0
    while left < len(ranges) - 1 and kept < edge_tokens:
        start, end = ranges[left]
        kept += count_tokens(history[start:end])
        left += 1
    right = len(ranges)
    kept = 0
    while right > left + 1 and kept < edge_tokens:
        right -= 1
        start, end = ranges[right]
        kept += count_tokens(history[start:end])
    if right <= left:
        return None
    middle_start = ranges[left][0]
    middle_end = ranges[right][0]
    summary = await summarize(settings, history[middle_start:middle_end])
    head = list(history[:middle_start])
    tail = list(history[middle_end:])
    # The middle starts at a fresh user turn, so the turn before it ends in a
    # ModelResponse. Attach the assistant-generated memory there.
    head[-1] = replace(head[-1], parts=[*head[-1].parts, checkpoint_part(summary)])
    return head + tail


class InputTooLarge(RuntimeError):
    """The largest turn is a single unanswered request -- nothing to shrink."""


async def collapse_largest_turn(settings: Settings, history: list[Any]) -> list[Any]:
    """Last resort: keep one turn's user request and replace all of its work."""
    ranges = turn_ranges(history)
    start, end = max(ranges, key=lambda r: count_tokens(history[r[0] : r[1]]))
    turn = history[start:end]
    if not any(isinstance(message, ModelResponse) for message in turn):
        # A turn this fat with no ModelResponse in it is just the raw request
        # -- there is no generated content to summarize away, so every other
        # strategy already failed and this one would only bolt a checkpoint
        # onto an unanswered request, silently swapping it for a fresh prompt
        # on resume. Say so and stop instead of pretending we fixed it.
        raise InputTooLarge("Input way too long, autocompact won't help.")
    summary = await summarize(settings, turn)
    collapsed = [turn[0], ModelResponse(parts=[checkpoint_part(summary)])]
    return history[:start] + collapsed + history[end:]


async def compact(settings: Settings, history: list[Any], recoveries: int) -> list[Any]:
    """Shrink one history, preserving as much recent detail as still fits.

    The verbatim edges are sized from the history being compacted -- roughly
    half of it survives a first compaction -- instead of a fixed few thousand
    tokens. A fixed budget turned a 90k-token all-night session into 8k in
    one step; proportional edges turn it into ~45k, and the escalation in
    ``run_turn.recover`` still halves them whenever that proves too generous.
    """
    history = strip_images(list(history))
    # A truncated trailing response is the thing that just failed. Dropping it
    # leaves the history ending in tool results, which lets the model resume
    # generating from exactly where it ran out of room.
    while history and isinstance(history[-1], ModelResponse):
        history.pop()

    before = count_tokens(history)
    edge_tokens = max(
        MIN_EDGE_TOKENS,
        min(
            int(before * COMPACT_KEEP_FRACTION) // 2,
            int(settings.context_window * EDGE_WINDOW_FRACTION),
        )
        // (2**recoveries),
    )

    # First: kill fat turns. One completed 30k-token coding turn must not
    # permanently consume 30k tokens of every future request.
    changed = False
    for start, end in reversed(turn_ranges(history)):
        turn = history[start:end]
        if count_tokens(turn) <= settings.context_window * FAT_TURN_FRACTION:
            continue
        compacted = await compact_one_turn(settings, turn, edge_tokens)
        if compacted is None:
            continue
        history[start:end] = compacted
        changed = True
    if changed:
        return history

    # No single fat turn: preserve both conversation edges and summarize the
    # complete turns in between.
    compacted = await compact_middle_turns(settings, history, edge_tokens)
    if compacted is not None:
        return compacted

    # Nothing clever left. Collapse the fattest turn down to its request.
    return await collapse_largest_turn(settings, history)


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
    * a response truncated with context to spare -> keep it and continue;
    * the model stuck on its own output (unparseable generation, or arguments
      it cannot repair) -> nudge it with a fresh user turn;
    * anything else -> say so, wait, and retry the same turn.

    The agent must already be entered (``async with agent``) so MCP servers
    stay connected across retries instead of reconnecting on every hiccup.
    """
    history = session.load_messages()
    next_prompt: str | None = prompt
    recoveries = 0
    floor: int | None = None  # size the previous compaction reached
    started = time.perf_counter()

    async def recover(reason: str, *, force_compact: bool) -> str | None:
        """Make room, then say what to send next.

        ``force_compact`` is set when the endpoint itself rejected the request
        for length -- our token estimate does not get a vote against that.
        Otherwise a truncated response with the context still mostly empty
        just means the answer outgrew --max-tokens, and summarizing would
        discard detail to solve a problem summarizing cannot solve.
        """
        nonlocal history, recoveries, floor
        before = count_tokens(history)
        if not force_compact and before < settings.context_window * COMPACT_ABOVE:
            history = drop_unanswered_tail(history)
            session.save_messages(history)
            note(f"{reason}; ~{before:,} tokens of {settings.context_window:,} used, "
                 "so this is the response limit, not the context. Continuing.")
            return resume_prompt(history, TRUNCATION_FEEDBACK)
        # Escalate only while stuck. Coming back here having barely grown
        # since the last compaction means that level of detail was still too
        # expensive, so halve the verbatim edges; coming back after real work
        # means it was affordable, so start over at full detail. Without the
        # reset a session that compacts all night would pin itself to the
        # minimum edges forever and throw away far more than it needs to.
        stuck = floor is not None and before <= floor + PROGRESS_TOKENS
        recoveries = recoveries + 1 if stuck else 0
        note(f"{reason}; compacting ~{before:,} tokens (recovery level {recoveries})")
        history = await compact(settings, history, recoveries)
        floor = count_tokens(history)
        session.save_messages(history)
        if floor >= before:
            # Already as small as this strategy can make it. Retrying
            # immediately would just summarize the same messages forever.
            note(f"compaction bottomed out at ~{floor:,} tokens; waiting")
            await asyncio.sleep(RETRY_DELAY_SECONDS)
        else:
            note(f"compacted to ~{floor:,} tokens; resuming")
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
                    # The endpoint refusing the request outranks our estimate.
                    force_compact=is_context_failure(exc),
                )
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
                "response hit the generation limit", force_compact=False
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
        f"  temperature:    {settings.temperature}",
        f"  max tokens:     {settings.max_tokens or 'server default'}",
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
    agent = build_agent(settings, discovery, store)
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
    agent = build_agent(
        settings, discovery, store,
        bash_machine=bash_machine,
        bash_machine_user=user,
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
        raise SystemExit(130) from None
    except InputTooLarge as exc:
        note(str(exc))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
