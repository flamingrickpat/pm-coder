r"""Local coding agent with interactive, resumable, and auto execution.
The upstream shape is intentionally recognizable: OpenAI-compatible
Pydantic AI, MCP discovery, project instructions, skills, and one host-shell
tool. PowerShell and Bash are small platform backends behind the same agent
logic. Sessions preserve structured model messages for exact conversational
continuation, while request and wall-clock limits remain explicit opt-ins.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import textwrap
import traceback
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any, Literal, cast
from xml.sax.saxutils import escape, quoteattr

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import (
    Agent,
    ModelRetry,
    RunContext,
    Tool,
    UsageLimits,
    capture_run_messages,
)
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UsageLimitExceeded,
)
from pydantic_ai.mcp import load_mcp_toolsets
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
)
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets import (
    ToolsetTool,
    WrapperToolset,
)
from pydantic_core import to_jsonable_python
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import FoldedScalarString, LiteralScalarString


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )


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
                if attempt == 3:
                    raise
                time.sleep(0.01 * (2**attempt))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def encode_exact_strings(value: Any) -> Any:
    """Keep exact text reconstructable while bounding physical YAML lines."""
    if isinstance(value, Mapping):
        return {str(key): encode_exact_strings(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [encode_exact_strings(item) for item in value]
    if not isinstance(value, str):
        return value
    if "\n" not in value and "\r" not in value and len(value) <= 48:
        return value
    encoded = json.dumps(value, ensure_ascii=False)
    return {
        "schema": "cog.exact-text.v1",
        "json_chunks": [encoded[index : index + 32] for index in range(0, len(encoded), 32)],
    }


def decode_exact_strings(value: Any) -> Any:
    """Reverse :func:`encode_exact_strings` with strict chunk validation."""
    if isinstance(value, Mapping):
        if value.get("schema") == "cog.exact-text.v1":
            chunks = value.get("json_chunks")
            if not isinstance(chunks, Sequence) or isinstance(chunks, str):
                raise ValueError("invalid cog.exact-text.v1 chunks")
            if not all(isinstance(chunk, str) for chunk in chunks):
                raise ValueError("invalid cog.exact-text.v1 chunks")
            return json.loads("".join(chunks))
        return {str(key): decode_exact_strings(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [decode_exact_strings(item) for item in value]
    return value


def _plain_data(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _plain_data(value.model_dump(mode="json", by_alias=True))
    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [_plain_data(item) for item in value]
    return value


def _readable_scalars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _readable_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_readable_scalars(item) for item in value]
    if not isinstance(value, str):
        return value
    if "\n" in value or (len(value) > 48 and any(character.isspace() for character in value)):
        return FoldedScalarString(value)
    return value


def _yaml(*, explicit_start: bool) -> YAML:
    yaml = YAML(typ="safe")
    yaml.allow_unicode = True
    yaml.default_flow_style = False
    yaml.explicit_start = explicit_start
    yaml.indent(mapping=2, sequence=4, offset=2)
    cast(Any, yaml).sort_base_mapping_type_on_output = False
    yaml.width = 64
    yaml.representer.add_representer(
        FoldedScalarString,
        lambda representer, data: representer.represent_scalar(
            "tag:yaml.org,2002:str", str(data), style=">"
        ),
    )
    return yaml


def _loader() -> YAML:
    yaml = YAML(typ="safe", pure=True)
    yaml.allow_duplicate_keys = False
    return yaml


def readable_yaml_bytes(value: Any, *, explicit_start: bool = False) -> bytes:
    stream = StringIO()
    _yaml(explicit_start=explicit_start).dump(_readable_scalars(_plain_data(value)), stream)
    return stream.getvalue().encode("utf-8")


def _human_message_scalars(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _human_message_scalars(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_human_message_scalars(item, key=key) for item in value]
    if not isinstance(value, str):
        return value
    markdown_start = value.lstrip().startswith(("#", "```", "- ", "> "))
    if "\n" in value or (key in {"content", "prompt"} and markdown_start):
        return LiteralScalarString(value)
    if len(value) > 120 and any(character.isspace() for character in value):
        return FoldedScalarString(value)
    return value


def readable_messages_yaml_bytes(messages: list[Any]) -> bytes:
    """Render model messages as a readable, non-canonical YAML companion."""
    yaml = YAML(typ="safe")
    yaml.allow_unicode = True
    yaml.default_flow_style = False
    yaml.width = 120
    yaml.indent(mapping=2, sequence=4, offset=2)
    cast(Any, yaml).sort_base_mapping_type_on_output = False
    yaml.representer.add_representer(
        LiteralScalarString,
        lambda representer, data: representer.represent_scalar(
            "tag:yaml.org,2002:str", str(data), style="|"
        ),
    )
    yaml.representer.add_representer(
        FoldedScalarString,
        lambda representer, data: representer.represent_scalar(
            "tag:yaml.org,2002:str", str(data), style=">"
        ),
    )
    stream = StringIO()
    yaml.dump(_human_message_scalars(to_jsonable_python(messages)), stream)
    return stream.getvalue().encode("utf-8")


def render_messages_yaml(
    messages_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Render an existing canonical ``messages.json`` file as readable YAML."""
    source = Path(messages_path).expanduser().resolve()
    target = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source.with_name("messages.yaml")
    )
    messages = ModelMessagesTypeAdapter.validate_json(source.read_bytes())
    atomic_write_bytes(target, readable_messages_yaml_bytes(messages))
    return target


def load_yaml_bytes(raw: bytes) -> Any:
    return _loader().load(raw.decode("utf-8"))


def wrap_markdown(text: str, *, width: int = 80) -> str:
    """Wrap prose while retaining Markdown headings, lists, fences, and blanks."""
    wrapped: list[str] = []
    in_fence = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            wrapped.append(raw_line)
            continue
        if in_fence or not stripped or stripped.startswith("#"):
            wrapped.append(raw_line)
            continue
        prefix = ""
        content = stripped
        if stripped.startswith("- "):
            prefix, content = "- ", stripped[2:]
        elif stripped.startswith("> "):
            prefix, content = "> ", stripped[2:]
        parts = textwrap.wrap(
            content,
            width=max(20, width - len(prefix)),
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        wrapped.append(prefix + parts[0] if parts else prefix.rstrip())
        wrapped.extend(" " * len(prefix) + part for part in parts[1:])
    return "\n".join(wrapped).rstrip() + "\n"

APP_NAME = "private-machine-coder"
DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_SHELL_TIMEOUT = 180
# Compatibility for callers that imported the old public constant.
DEFAULT_POWERSHELL_TIMEOUT = DEFAULT_SHELL_TIMEOUT
# None leaves discovery and tool-output dimensions unbounded. The response
# limit is deliberately finite: llama.cpp can otherwise spend the complete
# context on one malformed tool-call payload. -1 remains a legacy opt-out.
DEFAULT_MAX_TOOL_OUTPUT_CHARS: int | None = None
DEFAULT_MAX_SKILL_INDEX_CHARS: int | None = None
DEFAULT_MAX_PROJECT_INSTRUCTIONS_CHARS: int | None = None
DEFAULT_MAX_TOKENS: int | None = 8_192
DEFAULT_LOG_ROOT = Path("~/.pm/pm-coder").expanduser()
RECOVERY_TOOL_CALL_TOKEN_TARGET = 4_000
MAX_TRANSIENT_RETRY_DELAY_SECONDS = 30.0
MCP_CONFIG_CANDIDATES = (
    ".mcp.json",
    "mcp.json",
    "mcp_config.json",
    ".pi/mcp.json",
    ".codex/mcp.json",
)
FRONTMATTER_RE = re.compile(
    r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)",
    re.DOTALL,
)


class Settings(StrictModel):
    cwd: Path
    base_url: str
    api_key: str
    model: str
    mcp_config: Path | None
    shell_kind: Literal["powershell", "bash"]
    shell_executable: str
    shell_timeout: int = Field(gt=0)
    max_tool_output_chars: int | None = Field(default=None, gt=0)
    max_skill_index_chars: int | None = Field(default=None, gt=0)
    max_project_instructions_chars: int | None = Field(default=None, gt=0)
    temperature: float
    max_tokens: int | None = DEFAULT_MAX_TOKENS
    disable_thinking: bool = False

    @model_validator(mode="after")
    def paths_exist(self) -> Settings:
        if self.max_tokens is not None and (
            self.max_tokens < -1 or self.max_tokens == 0
        ):
            raise ValueError(
                "max_tokens must be a positive limit or -1"
            )
        if not self.cwd.is_dir():
            raise ValueError(f"working directory does not exist: {self.cwd}")
        if self.mcp_config is not None and not self.mcp_config.is_file():
            raise ValueError(f"MCP config does not exist: {self.mcp_config}")
        return self


class OneShotOptions(StrictModel):
    prompt_file: Path
    artifact_dir: Path
    request_limit: int = Field(gt=0)
    wall_clock_limit_seconds: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_files(self) -> OneShotOptions:
        if not self.prompt_file.is_file():
            raise ValueError(f"prompt file does not exist: {self.prompt_file}")
        return self


@dataclass(frozen=True)
class AutoResult:
    """Result returned by :func:`run_auto` and emitted by auto CLI mode."""

    response: str
    run_id: str
    duration_seconds: float
    tokens_used: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "response": self.response,
            "run_id": self.run_id,
            "duration_seconds": self.duration_seconds,
            "tokens_used": self.tokens_used,
        }


class SessionStore:
    """Persist one conversation as Pydantic AI model messages.

    ``messages.json`` contains the structured messages passed to
    ``Agent.run(message_history=...)``. Loading it does not summarize or
    re-prompt the conversation. With the same model, tokenizer, system
    instructions, tool definitions, and chat-template settings, the server
    receives the same logical prompt tokens as the previous turn. The
    server's private KV-cache memory is not serialized because llama.cpp does
    not expose a portable cache format through the OpenAI API.
    """

    schema = "pm-coder-session.v1"

    def __init__(self, path: Path, *, cwd: Path) -> None:
        self.path = path
        self.cwd = cwd.resolve()
        self.metadata_path = path / "session.json"
        self.messages_path = path / "messages.json"
        self.human_messages_path = path / "messages.yaml"
        self.runs_path = path / "runs.jsonl"
        self.path.mkdir(parents=True, exist_ok=True)

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
            if (
                not run_id
                or Path(run_id).name != run_id
                or run_id in {".", ".."}
            ):
                raise ValueError("run_id must be a single safe session directory name")
            path = root / run_id
        store = cls(path, cwd=cwd)
        if not store.metadata_path.exists():
            store._write_metadata(
                {
                    "schema": cls.schema,
                    "run_id": store.run_id,
                    "created_at": utc_now(),
                    "cwd": str(cwd.resolve()),
                    "message_history_format": "pydantic-ai-model-messages.v1",
                }
            )
        if store.messages_path.exists() and not store.human_messages_path.exists():
            render_messages_yaml(store.messages_path, store.human_messages_path)
        return store

    def _write_metadata(self, value: dict[str, Any]) -> None:
        atomic_write_bytes(
            self.metadata_path,
            json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"),
        )

    def load_messages(self) -> list[Any]:
        if not self.messages_path.exists():
            return []
        return ModelMessagesTypeAdapter.validate_json(self.messages_path.read_bytes())

    def save_messages(self, messages: list[Any]) -> None:
        payload = to_jsonable_python(messages)
        atomic_write_bytes(
            self.messages_path,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        )
        atomic_write_bytes(
            self.human_messages_path,
            readable_messages_yaml_bytes(messages),
        )

    def clear(self) -> None:
        self.save_messages([])

    def append_run(self, value: dict[str, Any]) -> None:
        with self.runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _prompt_text(prompt_or_path: str | Path, *, cwd: Path) -> str:
    value = Path(prompt_or_path).expanduser()
    if not value.is_absolute():
        value = cwd / value
    if value.is_file():
        return value.read_text(encoding="utf-8")
    return str(prompt_or_path)


class TruncatedModelOutputError(RuntimeError):
    """The model exhausted its response budget instead of completing."""


class ContextLimitReachedError(TruncatedModelOutputError):
    """The model filled the literal context before completing."""


class ToolCallParseError(RuntimeError):
    """The provider rejected a malformed generated tool-call payload."""


class EndpointRequestError(RuntimeError):
    """The endpoint rejected a request that retry feedback cannot repair."""


@dataclass(frozen=True)
class PlanCheckpoint:
    """The assigned plan state captured before a one-shot worker starts."""

    path: Path
    initial_sha256: str

    def satisfied(self) -> bool:
        try:
            current = sha256(self.path.read_bytes()).hexdigest()
        except OSError:
            return False
        return current != self.initial_sha256


@dataclass
class PlanCheckpointToolset(WrapperToolset[Any]):
    """Require a saved plan checkpoint before physical Minecraft effects."""

    checkpoint: PlanCheckpoint

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        if (
            _is_state_changing_minecraft_tool(name)
            and not self.checkpoint.satisfied()
        ):
            raise ModelRetry(
                "Before a state-changing Minecraft call, save a short "
                "dependency plan to the assigned plan file using the fresh "
                "typed observation. Then retry only after the plan file has "
                "actually changed."
            )
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)


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
        # Do not enable `set -e`: the tool must report the script's real exit
        # code and any diagnostics instead of changing ordinary shell semantics.
        return "set -o pipefail\n"

    def invocation(self, script_path: str) -> list[str]:
        return [self.executable, "--noprofile", "--norc", script_path]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def env_optional_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw)
    return None if value <= 0 else value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local coding agent with interactive and JSON-producing auto modes."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("interactive", "auto"),
        default="interactive",
        help="Run a persistent chat session or one JSON-producing prompt.",
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
    parser.add_argument(
        "--run-id",
        help="Resume this session directory under ~/.pm/pm-coder.",
    )
    parser.add_argument(
        "--log-root",
        default=str(DEFAULT_LOG_ROOT),
        help="Override the session log directory (mainly useful for tests).",
    )
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument(
        "--base-url",
        default=env_first("LOCAL_AGENT_BASE_URL", "OPENAI_BASE_URL")
        or DEFAULT_BASE_URL,
    )
    parser.add_argument(
        "--api-key",
        default=env_first("LOCAL_AGENT_API_KEY", "OPENAI_API_KEY") or "local",
    )
    parser.add_argument(
        "--model",
        default=env_first("LOCAL_AGENT_MODEL", "OPENAI_MODEL"),
    )
    parser.add_argument("--mcp-config")
    parser.add_argument(
        "--shell",
        choices=("auto", "powershell", "bash"),
        default=os.environ.get("LOCAL_AGENT_SHELL", "auto"),
        help="Host shell tool. The default selects PowerShell on Windows and Bash elsewhere.",
    )
    parser.add_argument(
        "--shell-timeout",
        "--powershell-timeout",
        dest="shell_timeout",
        type=int,
        default=int(
            os.environ.get(
                "LOCAL_AGENT_SHELL_TIMEOUT",
                os.environ.get(
                    "LOCAL_AGENT_POWERSHELL_TIMEOUT",
                    DEFAULT_SHELL_TIMEOUT,
                ),
            )
        ),
        help="Seconds allowed for one host-shell tool call.",
    )
    parser.add_argument(
        "--max-tool-output",
        type=int,
        default=env_optional_int(
            "LOCAL_AGENT_MAX_TOOL_OUTPUT", DEFAULT_MAX_TOOL_OUTPUT_CHARS
        ),
        help="Maximum shell-tool output characters; omitted means unlimited.",
    )
    parser.add_argument(
        "--max-skill-index",
        type=int,
        default=env_optional_int(
            "LOCAL_AGENT_MAX_SKILL_INDEX", DEFAULT_MAX_SKILL_INDEX_CHARS
        ),
        help="Maximum skill-index characters; omitted means unlimited.",
    )
    parser.add_argument(
        "--max-project-instructions",
        type=int,
        default=env_optional_int(
            "LOCAL_AGENT_MAX_PROJECT_INSTRUCTIONS",
            DEFAULT_MAX_PROJECT_INSTRUCTIONS_CHARS,
        ),
        help="Maximum instruction characters; omitted means unlimited.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("LOCAL_AGENT_TEMPERATURE", "0.1")),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=env_optional_int("LOCAL_AGENT_MAX_TOKENS", DEFAULT_MAX_TOKENS),
        help=(
            "Generated tokens per model response. Omitted means the server "
            "chooses its limit; use a positive value to override it."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=True,
        help="Allow provider-specific long-form reasoning (the default).",
    )
    parser.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable provider-specific long-form reasoning.",
    )
    parser.add_argument("--prompt-file")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--request-limit", type=int)
    parser.add_argument("--wall-clock-limit", type=int)
    return parser.parse_args(argv)


def discover_model_id(base_url: str, api_key: str) -> str | None:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    for model in payload.get("data", []):
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id:
            return model_id
    return None


def find_powershell() -> str:
    for executable in ("pwsh.exe", "pwsh", "powershell.exe", "powershell"):
        resolved = shutil.which(executable)
        if resolved:
            return resolved
    raise RuntimeError("PowerShell is required by the minimal agent")


def find_bash() -> str:
    resolved = shutil.which("bash")
    if resolved:
        return resolved
    raise RuntimeError("Bash is required by the minimal agent")


def select_shell(requested: str = "auto") -> ShellBackend:
    """Resolve one platform backend while keeping the agent itself shared."""
    kind = requested
    if kind == "auto":
        kind = "powershell" if os.name == "nt" else "bash"
    if kind == "powershell":
        return PowerShellBackend(find_powershell())
    if kind == "bash":
        return BashBackend(find_bash())
    raise ValueError(f"unsupported shell: {requested}")


def shell_backend(settings: Settings) -> ShellBackend:
    if settings.shell_kind == "powershell":
        return PowerShellBackend(settings.shell_executable)
    return BashBackend(settings.shell_executable)


def ancestor_directories_nearest_first(cwd: Path) -> list[Path]:
    directories: list[Path] = []
    current = cwd.resolve()
    while True:
        directories.append(current)
        if current.parent == current:
            return directories
        current = current.parent


def find_mcp_config(cwd: Path, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = cwd / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MCP config does not exist: {path}")
        return path
    for directory in ancestor_directories_nearest_first(cwd):
        for relative_name in MCP_CONFIG_CANDIDATES:
            candidate = directory / relative_name
            if candidate.is_file():
                return candidate.resolve()
    return None


def read_mcp_server_names(config_path: Path | None) -> list[str]:
    if config_path is None:
        return []
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        raise ValueError(
            "MCP config must contain a top-level mcpServers object: "
            f"{config_path}"
        )
    return [str(name) for name in servers]


def skill_roots(cwd: Path) -> list[Path]:
    roots: list[Path] = []
    for directory in ancestor_directories_nearest_first(cwd):
        roots.extend(
            [
                directory / ".agents" / "skills",
                directory / ".pi" / "skills",
                directory / ".codex" / "skills",
            ]
        )
    roots.extend(
        [
            Path.home() / ".agents" / "skills",
            Path.home() / ".pi" / "agent" / "skills",
            Path.home() / ".codex" / "skills",
        ]
    )
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
        loaded = load_yaml_bytes(match.group(1).encode("utf-8"))
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError("YAML frontmatter must be an object")
        metadata = loaded or {}
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
    return Skill(
        name=name,
        description=description[:800],
        skill_file=skill_file.resolve(),
        priority=priority,
    )


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
    return (
        sorted(
            skills_by_name.values(),
            key=lambda skill: (skill.priority, skill.name.casefold()),
        ),
        errors,
    )


def format_skill_index(skills: list[Skill], max_chars: int | None) -> str:
    if not skills:
        return "<available_skills />"
    opening = "<available_skills>\n"
    closing = "</available_skills>"
    chunks = [opening]
    current_length = len(opening) + len(closing)
    omitted = 0
    for skill in skills:
        chunk = (
            "  <skill>\n"
            f"    <name>{escape(skill.name)}</name>\n"
            f"    <description>{escape(skill.description)}</description>\n"
            f"    <location>{escape(str(skill.skill_file))}</location>\n"
            "  </skill>\n"
        )
        if max_chars is not None and current_length + len(chunk) > max_chars:
            omitted += 1
            continue
        chunks.append(chunk)
        current_length += len(chunk)
    if omitted:
        chunks.append(
            f"  <omitted count={quoteattr(str(omitted))}>"
            "Additional skills were omitted from the initial context."
            "</omitted>\n"
        )
    chunks.append(closing)
    return "".join(chunks)


def discover_instruction_files(cwd: Path) -> list[Path]:
    files: list[Path] = []
    for directory in reversed(ancestor_directories_nearest_first(cwd)):
        for filename in ("AGENTS.md", "CLAUDE.md"):
            candidate = directory / filename
            if candidate.is_file() and candidate.resolve() not in files:
                files.append(candidate.resolve())
    return files


def load_project_instructions(files: list[Path], max_chars: int | None) -> str:
    if not files:
        return "<project_instructions />"
    opening = "<project_instructions>\n"
    closing = "</project_instructions>"
    chunks = [opening]
    current_length = len(opening) + len(closing)
    omitted = 0
    for path in files:
        content = path.read_text(encoding="utf-8")
        chunk = (
            f"  <instruction_file path={quoteattr(str(path))}>\n"
            f"{content}\n"
            "  </instruction_file>\n"
        )
        if max_chars is not None and current_length + len(chunk) > max_chars:
            omitted += 1
            continue
        chunks.append(chunk)
        current_length += len(chunk)
    if omitted:
        chunks.append(
            f"  <omitted count={quoteattr(str(omitted))}>"
            "Some instruction files exceeded the context budget."
            "</omitted>\n"
        )
    chunks.append(closing)
    return "".join(chunks)


def truncate_middle(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    marker = f"\n\n... [{len(text) - max_chars:,} characters omitted] ...\n\n"
    remaining = max_chars - len(marker)
    head = max(0, remaining // 2)
    return text[:head] + marker + text[-(remaining - head) :]


def make_shell_tool(settings: Settings) -> Tool[Any]:
    backend = shell_backend(settings)

    def host_shell(
        command: str,
        timeout_seconds: int = settings.shell_timeout,
    ) -> str:
        """Execute a host-shell script in the selected agent workspace."""
        print(f"\n[{backend.kind}]", flush=True)
        print(command.rstrip(), flush=True)
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
            result = (
                f"exit_code: {completed.returncode}\n"
                f"stdout:\n{stdout or '(empty)'}\n"
                f"stderr:\n{stderr or '(empty)'}"
            )
            print(f"[{backend.kind} exit {completed.returncode}]", flush=True)
            return truncate_middle(result, settings.max_tool_output_chars)
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
            return truncate_middle(
                (
                    "timed_out: true\n"
                    f"timeout_seconds: {timeout_seconds}\n"
                    f"stdout_before_timeout:\n{stdout or '(empty)'}\n"
                    f"stderr_before_timeout:\n{stderr or '(empty)'}"
                ),
                settings.max_tool_output_chars,
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


def build_system_prompt(
    settings: Settings,
    skill_index: str,
    project_instructions: str,
    mcp_server_names: list[str],
) -> str:
    mcp_summary = ", ".join(mcp_server_names) or "none configured"
    shell = shell_backend(settings)
    minecraft_guidance = (
        "\nMCP tools are exposed with server prefixes. Observe Minecraft before "
        "acting and use ordinary survival mechanics only. A process exit is not "
        "proof of game success; the coordinator owns final postconditions.\n"
        if any("minecraft" in name.casefold() for name in mcp_server_names)
        else "\nMCP tools are exposed with server prefixes. Use them when the "
        "assigned workflow or project instructions require them.\n"
    )
    return textwrap.dedent(
        f"""
        You are a local coding agent operating directly in one
        agent workspace.

        <environment>
          <working_directory>{escape(str(settings.cwd))}</working_directory>
          <host_shell name={quoteattr(shell.kind)}>
            {escape(shell.executable)}
          </host_shell>
          <mcp_servers>{escape(mcp_summary)}</mcp_servers>
        </environment>

        Complete the user's current request. Inspect real state, make
        concrete progress, verify every claimed effect, and leave durable
        evidence when useful. Conversation history may continue across
        requests, so use it as context without repeating completed work.

        `{shell.kind}` starts in the selected workspace. Use it for local file
        inspection, edits, and verification. Project instructions define the
        exact writable paths. Do not escape the selected workspace.
        {minecraft_guidance}

        Skills are reusable workflows. If one clearly applies, read its full
        SKILL.md before using it.

        {skill_index}

        Apply these discovered project instructions:

        {project_instructions}
        """
    ).strip()


def build_settings(args: argparse.Namespace) -> Settings:
    cwd = Path(args.cwd).expanduser().resolve()
    base_url = args.base_url.rstrip("/")
    model = args.model or discover_model_id(base_url, args.api_key) or "local"
    backend = select_shell(args.shell)
    return Settings(
        cwd=cwd,
        base_url=base_url,
        api_key=args.api_key,
        model=model,
        mcp_config=find_mcp_config(cwd, args.mcp_config),
        shell_kind=backend.kind,
        shell_executable=backend.executable,
        shell_timeout=args.shell_timeout,
        max_tool_output_chars=args.max_tool_output,
        max_skill_index_chars=args.max_skill_index,
        max_project_instructions_chars=args.max_project_instructions,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        disable_thinking=not args.enable_thinking,
    )


def discover_workspace(settings: Settings) -> DiscoveryResult:
    skills, skill_errors = load_skills(settings.cwd)
    instruction_files = discover_instruction_files(settings.cwd)
    return DiscoveryResult(
        skills=skills,
        skill_errors=skill_errors,
        instruction_files=instruction_files,
        project_instructions=load_project_instructions(
            instruction_files,
            settings.max_project_instructions_chars,
        ),
        mcp_server_names=read_mcp_server_names(settings.mcp_config),
    )


def build_agent(
    settings: Settings,
    discovery: DiscoveryResult,
    *,
    plan_checkpoint: PlanCheckpoint | None = None,
) -> Agent[Any, str]:
    profile = OpenAIModelProfile(
        openai_supports_strict_tool_definition=False,
        openai_chat_supports_multiple_system_messages=False,
    )
    model = OpenAIChatModel(
        settings.model,
        provider=OpenAIProvider(
            base_url=settings.base_url,
            api_key=settings.api_key,
        ),
        profile=profile,
    )
    toolsets = (
        load_mcp_toolsets(settings.mcp_config)
        if settings.mcp_config is not None
        else []
    )
    if plan_checkpoint is not None:
        toolsets = [
            PlanCheckpointToolset(
                wrapped=toolset,
                checkpoint=plan_checkpoint,
            )
            for toolset in toolsets
        ]
    return Agent(
        model=model,
        instructions=build_system_prompt(
            settings,
            format_skill_index(
                discovery.skills,
                settings.max_skill_index_chars,
            ),
            discovery.project_instructions,
            discovery.mcp_server_names,
        ),
        tools=[make_shell_tool(settings)],
        toolsets=toolsets,
        model_settings=OpenAIChatModelSettings(
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            parallel_tool_calls=False,
            extra_body=(
                {"chat_template_kwargs": {"enable_thinking": False}}
                if settings.disable_thinking
                else {}
            ),
        ),
        retries=3,
        max_concurrency=1,
    )


def one_shot_options(args: argparse.Namespace) -> OneShotOptions | None:
    # The old heartbeat mode is selected by --artifact-dir. In the new auto
    # mode --prompt-file is a normal prompt source and must not activate it.
    if args.artifact_dir is None:
        return None
    values = (
        args.prompt_file,
        args.artifact_dir,
        args.request_limit,
        args.wall_clock_limit,
    )
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise ValueError(
            "one-shot mode requires --prompt-file, --artifact-dir, "
            "--request-limit, and --wall-clock-limit"
        )
    return OneShotOptions(
        prompt_file=Path(args.prompt_file).expanduser().resolve(),
        artifact_dir=Path(args.artifact_dir).expanduser().resolve(),
        request_limit=args.request_limit,
        wall_clock_limit_seconds=args.wall_clock_limit,
    )


def plan_checkpoint_for_one_shot(
    settings: Settings,
    options: OneShotOptions,
) -> PlanCheckpoint | None:
    """Capture the assigned plan named by a generated heartbeat prompt."""
    prompt = options.prompt_file.read_text(encoding="utf-8")
    match = re.search(
        r"(?m)^assigned_plan_file:\s*(?:\r?\n[ \t]+)?"
        r"(?P<path>\S+\.plan\.md)\s*$",
        prompt,
    )
    if match is None:
        return None
    path = (settings.cwd / match.group("path")).resolve()
    try:
        relative = path.relative_to(settings.cwd.resolve())
    except ValueError as exc:
        raise ValueError(
            "assigned plan file is outside the agent workspace"
        ) from exc
    if not relative.parts or relative.parts[0] != "tasks":
        raise ValueError("assigned plan file must be under tasks/")
    if not path.is_file():
        raise ValueError(f"assigned plan file does not exist: {relative}")
    return PlanCheckpoint(
        path=path,
        initial_sha256=sha256(path.read_bytes()).hexdigest(),
    )


def _is_state_changing_minecraft_tool(name: str) -> bool:
    return any(
        name.endswith(suffix)
        for suffix in (
            "minecraft_call",
            "minecraft_walk_to",
            "minecraft_mine_block",
            "minecraft_pillar_up",
            "minecraft_collect_blocks",
            "minecraft_craft_item",
            "minecraft_smelt_item",
            "minecraft_equip",
            "minecraft_rotate",
            "minecraft_execute_typescript",
            "minecraft_suicide",
            "minecraft_retire_character",
        )
    )


def _write_yaml(path: Path, value: Any) -> None:
    atomic_write_bytes(path, readable_yaml_bytes(encode_exact_strings(value)))


def _bounded_markdown(value: str) -> str:
    lines: list[str] = []
    for line in wrap_markdown(value, width=72).splitlines():
        if len(line) <= 80:
            lines.append(line)
            continue
        lines.extend(
            line[index : index + 72]
            for index in range(0, len(line), 72)
        )
    return "\n".join(lines).rstrip() + "\n"


def _terminal_finish_reason(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("kind") != "response":
            continue
        finish_reason = message.get("finish_reason")
        if isinstance(finish_reason, str):
            return finish_reason
        provider_details = message.get("provider_details")
        if isinstance(provider_details, dict):
            provider_reason = provider_details.get("finish_reason")
            if isinstance(provider_reason, str):
                return provider_reason
        return None
    return None


def _last_response_usage(messages: list[Any]) -> dict[str, int]:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("kind") != "response":
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            return {}
        return {
            key: value
            for key, value in usage.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
    return {}


def _captured_usage(messages: list[Any]) -> dict[str, int]:
    totals: dict[str, int] = {"requests": 0}
    for message in messages:
        if not isinstance(message, dict) or message.get("kind") != "response":
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        totals["requests"] += 1
        for key, value in usage.items():
            if (
                key != "total_tokens"
                and isinstance(value, int)
                and not isinstance(value, bool)
            ):
                totals[key] = totals.get(key, 0) + value
        request_total = usage.get("total_tokens")
        if not isinstance(request_total, int):
            request_total = sum(
                value
                for key, value in usage.items()
                if key in {"input_tokens", "output_tokens"}
                and isinstance(value, int)
                and not isinstance(value, bool)
            )
        totals["total_tokens"] = (
            totals.get("total_tokens", 0) + request_total
        )
    return totals if totals["requests"] else {}


def _last_tool_call_summary(messages: list[Any]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        parts = message.get("parts")
        if not isinstance(parts, list):
            continue
        for part in reversed(parts):
            if not isinstance(part, dict):
                continue
            if (part.get("part_kind") or part.get("kind")) != "tool-call":
                continue
            name = part.get("tool_name")
            name = name if isinstance(name, str) and name else "unknown"
            args = part.get("args")
            command = args.get("command") if isinstance(args, dict) else None
            if isinstance(command, str) and command.strip():
                preview = " ".join(command.split())
                if len(preview) > 120:
                    preview = preview[:117] + "..."
                return f'tool call {name} "{preview}"'
            return f"tool call {name}"
    return "model response"


def _generation_boundary_failure(
    settings: Settings | int | None,
    messages: list[Any],
) -> tuple[type[TruncatedModelOutputError], str] | None:
    if _terminal_finish_reason(messages) != "length":
        return None
    usage = _last_response_usage(messages)
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = usage.get("input_tokens", 0) + usage.get(
            "output_tokens", 0
        )
    token_text = (
        f" after {total_tokens:,} tokens in the last recorded request"
        if total_tokens
        else ""
    )
    activity = _last_tool_call_summary(messages)
    advice = (
        "keep each tool-call payload below about 4,000 generated tokens and "
        "split large scripts, commands, or file edits into smaller calls"
    )
    max_tokens = settings.max_tokens if isinstance(settings, Settings) else settings
    if max_tokens is None or max_tokens == -1:
        return (
            ContextLimitReachedError,
            f"{activity} reached the model context limit{token_text}; {advice}",
        )
    output_tokens = usage.get("output_tokens")
    if (
        isinstance(output_tokens, int)
        and output_tokens > 0
        and output_tokens < int(max_tokens * 0.9)
    ):
        return (
            ContextLimitReachedError,
            f"{activity} reached the model context limit{token_text} before "
            f"it could use the {max_tokens:,}-token response limit; {advice}",
        )
    return (
        TruncatedModelOutputError,
        f"{activity} reached the {max_tokens:,}-token response limit"
        f"{token_text}; {advice}",
    )


def _tool_call_parse_failure(exc: BaseException) -> str | None:
    raw = _exception_text(exc)
    lowered = raw.casefold()
    known_message = "failed to parse tool call arguments as json" in lowered
    generic_message = "tool call arguments" in lowered and any(
        marker in lowered
        for marker in ("invalid json", "json parse", "unterminated")
    )
    if not (known_message or generic_message):
        return None
    column_match = re.search(r"column (\d+)", raw)
    position = (
        f" after {int(column_match.group(1)):,} JSON characters"
        if column_match
        else ""
    )
    preview_match = re.search(r"last read:\s*(.*)", raw, re.DOTALL)
    preview = ""
    if preview_match:
        preview = " ".join(preview_match.group(1).split())
        preview = preview.replace("\\n", " ")
        if len(preview) > 140:
            preview = preview[:137] + "..."
    subject = (
        f'tool-call payload beginning "{preview}"'
        if preview
        else "tool-call payload"
    )
    return (
        f"{subject} was rejected{position} because its JSON was incomplete; "
        "the tool did not run"
    )


def _exception_text(exc: BaseException) -> str:
    """Return useful provider text without exposing an unbounded response body."""
    parts = [str(exc)]
    if isinstance(exc, ModelHTTPError) and exc.body is not None:
        try:
            parts.append(json.dumps(exc.body, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            parts.append(str(exc.body))
    return "\n".join(part for part in parts if part)


def _context_limit_failure(exc: BaseException) -> str | None:
    raw = " ".join(_exception_text(exc).split())
    lowered = raw.casefold()
    markers = (
        "context limit",
        "context window",
        "maximum context length",
        "exceeds the context",
        "exceeded the context",
        "prompt is too long",
        "request exceeds the available context",
        "n_ctx",
    )
    if not any(marker in lowered for marker in markers):
        return None
    preview = raw[:600] + ("..." if len(raw) > 600 else "")
    return (
        "The endpoint rejected the request because the conversation reached "
        f"the model context limit. Start a fresh kernel attempt. Provider: {preview}"
    )


def _is_transient_endpoint_failure(exc: BaseException) -> bool:
    if isinstance(exc, ModelHTTPError):
        return exc.status_code in {429, 500, 502, 503, 504}
    if isinstance(exc, ModelAPIError):
        return True
    for cause in _exception_chain(exc):
        name = type(cause).__name__.casefold()
        module = type(cause).__module__.casefold()
        if any(marker in name for marker in ("connection", "connect", "timeout")):
            return True
        if module.startswith(("httpx", "httpcore", "openai")) and isinstance(
            cause, (OSError, TimeoutError)
        ):
            return True
    return False


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _endpoint_failure(exc: BaseException) -> tuple[type[RuntimeError], str] | None:
    context = _context_limit_failure(exc)
    if context is not None:
        return ContextLimitReachedError, context
    if not isinstance(exc, ModelHTTPError):
        return None
    status = exc.status_code
    explanations = {
        400: "The endpoint rejected the request as invalid.",
        401: "The endpoint rejected the API credentials.",
        403: "The endpoint denied access to the model or operation.",
        404: "The endpoint or requested model was not found.",
        413: "The endpoint rejected the request because its payload was too large.",
        422: "The endpoint could not process the request payload.",
    }
    summary = explanations.get(status, f"The endpoint returned HTTP {status}.")
    raw = " ".join(_exception_text(exc).split())
    if len(raw) > 600:
        raw = raw[:597] + "..."
    return EndpointRequestError, f"{summary} Provider: {raw}"


def _safe_recovery_history(messages: list[Any], *, prefix_count: int) -> list[Any]:
    """Keep completed history and remove only the failed terminal exchange."""
    data = to_jsonable_python(messages)
    cutoff = len(messages)
    for index in range(prefix_count, len(messages)):
        item = data[index]
        if isinstance(item, dict) and item.get("state") == "interrupted":
            cutoff = index
            break
    for index in range(cutoff - 1, prefix_count - 1, -1):
        item = data[index]
        if (
            isinstance(item, dict)
            and item.get("kind") == "response"
            and _terminal_finish_reason([item]) == "length"
        ):
            cutoff = index
            break
    return list(messages[:cutoff])


def _transient_recovery_state(
    messages: list[Any],
    *,
    prefix_count: int,
    retry_prompt: str,
) -> tuple[list[Any], str]:
    """Retry an unanswered user prompt without adding it to the history."""
    history = _safe_recovery_history(messages, prefix_count=prefix_count)
    if len(history) <= prefix_count:
        return history, retry_prompt
    data = to_jsonable_python(history[-1])
    if not isinstance(data, dict) or data.get("kind") != "request":
        return history, _transient_retry_feedback()
    parts = data.get("parts")
    if not isinstance(parts, list) or not parts:
        return history, _transient_retry_feedback()
    kinds = {
        part.get("part_kind") or part.get("kind")
        for part in parts
        if isinstance(part, dict)
    }
    if kinds and kinds <= {"system-prompt", "user-prompt"}:
        return history[:-1], retry_prompt
    return history, _transient_retry_feedback()


def _merge_usage(total: dict[str, Any], addition: Mapping[str, Any]) -> None:
    for key, value in addition.items():
        if isinstance(value, Mapping):
            nested = total.setdefault(key, {})
            if isinstance(nested, dict):
                _merge_usage(nested, value)
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            total[key] = int(total.get(key, 0) or 0) + value
        elif key not in total:
            total[key] = value


def _response_limit_feedback(max_tokens: int | None) -> str:
    limit = f"{max_tokens:,}" if max_tokens not in {None, -1} else "the available"
    return (
        f"The previous response reached {limit} generated tokens and was discarded. "
        "Do not repeat or enlarge it. Inspect the current repository state. "
        f"Keep each tool-call payload below about {RECOVERY_TOOL_CALL_TOKEN_TARGET:,} "
        "generated tokens. Split a large script, command, or file edit into short "
        "sequential edits. Continue the assigned role and return its required result."
    )


def _tool_call_parse_feedback(detail: str) -> str:
    return (
        f"The endpoint rejected the previous response: {detail} "
        "Do not repeat or enlarge that tool call. Inspect the current repository state. "
        f"Keep each tool-call payload below about {RECOVERY_TOOL_CALL_TOKEN_TARGET:,} "
        "generated tokens. Create a small file skeleton first, then add sections with "
        "short sequential edits. Continue the assigned role and return its required result."
    )


def _transient_retry_feedback() -> str:
    return (
        "The model endpoint disconnected or restarted. Continue the assigned role "
        "from the current repository state. Do not repeat a tool call that already ran."
    )


def _remaining_wall_clock(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("the agent turn reached its wall-clock limit")
    return remaining


def _transient_retry_delay(exc: BaseException, consecutive: int) -> float:
    if isinstance(exc, ModelHTTPError) and exc.retry_after is not None:
        return min(MAX_TRANSIENT_RETRY_DELAY_SECONDS, max(0.0, exc.retry_after))
    return min(MAX_TRANSIENT_RETRY_DELAY_SECONDS, float(2 ** min(consecutive - 1, 5)))


def _workspace_path(path: Path, workspace: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _discovery_document(
    discovery: DiscoveryResult,
    workspace: Path,
) -> dict[str, Any]:
    return {
        "schema": "cog.minimal-agent-discovery.v1",
        "instruction_files": [
            _workspace_path(path, workspace)
            for path in discovery.instruction_files
        ],
        "mcp_servers": discovery.mcp_server_names,
        "skills": [
            {
                "name": skill.name,
                "description": skill.description,
                "skill_file": _workspace_path(skill.skill_file, workspace),
            }
            for skill in discovery.skills
        ],
        "skill_errors": discovery.skill_errors,
    }


async def run_one_shot(
    agent: Any,
    settings: Settings,
    discovery: DiscoveryResult,
    options: OneShotOptions,
) -> str:
    prompt_path = options.prompt_file.resolve()
    artifact_path = options.artifact_dir.resolve()
    try:
        prompt_relative = prompt_path.relative_to(settings.cwd.resolve())
        artifact_path.relative_to(settings.cwd.resolve() / "artifacts")
    except ValueError as exc:
        raise ValueError(
            "one-shot prompts and artifacts must stay in the agent workspace"
        ) from exc
    options.artifact_dir.mkdir(parents=True, exist_ok=False)
    prompt = options.prompt_file.read_text(encoding="utf-8")
    started_at = utc_now()
    _write_yaml(
        options.artifact_dir / "input.yaml",
        {
            "schema": "cog.minimal-agent-input.v1",
            "started_at": started_at,
            "cwd": ".",
            "model": settings.model,
            "base_url": settings.base_url,
            "mcp_config": (
                _workspace_path(settings.mcp_config, settings.cwd)
                if settings.mcp_config is not None
                else None
            ),
            "request_limit": options.request_limit,
            "wall_clock_limit_seconds": options.wall_clock_limit_seconds,
            "max_tokens_per_request": settings.max_tokens,
            "thinking_disabled": settings.disable_thinking,
            "prompt_file": prompt_relative.as_posix(),
            "prompt": prompt,
        },
    )
    _write_yaml(
        options.artifact_dir / "discovery.yaml",
        _discovery_document(discovery, settings.cwd),
    )
    captured_messages: list[Any]
    with capture_run_messages() as captured_messages:
        try:
            with Agent.parallel_tool_call_execution_mode("sequential"):
                async with agent:
                    result = await asyncio.wait_for(
                        agent.run(
                            prompt,
                            message_history=None,
                            usage_limits=UsageLimits(
                                request_limit=options.request_limit,
                            ),
                        ),
                        timeout=options.wall_clock_limit_seconds,
                    )
            messages = to_jsonable_python(result.all_messages())
            usage = asdict(result.usage)
            output = str(result.output)
            finish_reason = _terminal_finish_reason(messages)
            _write_yaml(
                options.artifact_dir / "transcript.yaml",
                {
                    "schema": "cog.minimal-agent-transcript.v1",
                    "messages": messages,
                    **(
                        {"incomplete": True}
                        if finish_reason == "length"
                        else {}
                    ),
                },
            )
            atomic_write_bytes(
                options.artifact_dir / "result.md",
                _bounded_markdown(
                    (
                        "# Worker incomplete output\n\n"
                        if finish_reason == "length"
                        else "# Worker result\n\n"
                    )
                    + output
                ).encode("utf-8"),
            )
            if finish_reason == "length":
                boundary_failure = _generation_boundary_failure(
                    settings,
                    messages,
                )
                assert boundary_failure is not None
                error_class, error = boundary_failure
                _write_yaml(
                    options.artifact_dir / "session.yaml",
                    {
                        "schema": "cog.minimal-agent-session.v1",
                        "status": "failed",
                        "started_at": started_at,
                        "finished_at": utc_now(),
                        "request_limit": options.request_limit,
                        "requests_used": usage.get("requests", 0),
                        "usage": usage,
                        "transcript": "transcript.yaml",
                        "result": "result.md",
                        "error_type": error_class.__name__,
                        "error": error,
                    },
                )
                raise error_class(error)
            _write_yaml(
                options.artifact_dir / "session.yaml",
                {
                    "schema": "cog.minimal-agent-session.v1",
                    "status": "succeeded",
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "request_limit": options.request_limit,
                    "requests_used": usage.get("requests", 0),
                    "usage": usage,
                    "transcript": "transcript.yaml",
                    "result": "result.md",
                },
            )
            return output
        except BaseException as exc:
            if isinstance(exc, TruncatedModelOutputError):
                raise
            messages = to_jsonable_python(captured_messages)
            usage = _captured_usage(messages)
            boundary_failure = _generation_boundary_failure(
                settings,
                messages,
            )
            parse_failure = _tool_call_parse_failure(exc)
            normalized_error_class: type[RuntimeError] | None = None
            normalized_error: str | None = None
            if boundary_failure is not None:
                normalized_error_class, normalized_error = boundary_failure
            elif parse_failure is not None:
                normalized_error_class = ToolCallParseError
                normalized_error = parse_failure
            _write_yaml(
                options.artifact_dir / "transcript.yaml",
                {
                    "schema": "cog.minimal-agent-transcript.v1",
                    "messages": messages,
                    "incomplete": True,
                },
            )
            _write_yaml(
                options.artifact_dir / "session.yaml",
                {
                    "schema": "cog.minimal-agent-session.v1",
                    "status": (
                        "timed_out"
                        if isinstance(exc, TimeoutError)
                        else "failed"
                    ),
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "request_limit": options.request_limit,
                    **(
                        {"requests_used": usage.get("requests", 0)}
                        if usage
                        else {}
                    ),
                    **({"usage": usage} if usage else {}),
                    "transcript": "transcript.yaml",
                    "error_type": (
                        normalized_error_class.__name__
                        if normalized_error_class is not None
                        else type(exc).__name__
                    ),
                    "error": normalized_error or str(exc),
                    **(
                        {
                            "cause_type": type(exc).__name__,
                            "cause": str(exc)[:4000],
                        }
                        if normalized_error is not None
                        else {}
                    ),
                    "traceback": "".join(
                        traceback.format_exception(exc)
                    )[-8000:],
                },
            )
            if (
                normalized_error_class is not None
                and normalized_error is not None
            ):
                raise normalized_error_class(normalized_error) from exc
            raise


def _usage_dict(usage: Any) -> dict[str, Any]:
    """Convert Pydantic AI usage into stable JSON data."""
    value = to_jsonable_python(asdict(usage))
    if isinstance(value, dict) and "total_tokens" not in value:
        input_tokens = value.get("input_tokens", 0) or 0
        output_tokens = value.get("output_tokens", 0) or 0
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            value["total_tokens"] = input_tokens + output_tokens
    return value


async def _run_agent_turn(
    agent: Agent[Any, str],
    prompt: str,
    message_history: list[Any] | None,
    *,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    request_limit: int | None = None,
    wall_clock_limit: int | None = None,
) -> tuple[Any, list[Any], dict[str, Any]]:
    history = list(message_history or [])
    next_prompt = prompt
    used_requests = 0
    total_usage: dict[str, Any] = {}
    transient_failures = 0
    deadline = (
        time.monotonic() + wall_clock_limit
        if wall_clock_limit is not None
        else None
    )

    while True:
        remaining_requests = (
            request_limit - used_requests if request_limit is not None else None
        )
        if remaining_requests is not None and remaining_requests <= 0:
            raise UsageLimitExceeded(
                f"The agent turn used its {request_limit} model requests"
            )
        limits = (
            UsageLimits(request_limit=remaining_requests)
            if remaining_requests is not None
            else None
        )
        captured: list[Any] = []
        try:
            with capture_run_messages() as captured:
                with Agent.parallel_tool_call_execution_mode("sequential"):
                    call = agent.run(
                        next_prompt,
                        message_history=history or None,
                        **({"usage_limits": limits} if limits is not None else {}),
                    )
                    timeout = _remaining_wall_clock(deadline)
                    result = await (
                        asyncio.wait_for(call, timeout=timeout)
                        if timeout is not None
                        else call
                    )
        except BaseException as exc:
            new_messages = captured[len(history) :]
            observed_usage = _captured_usage(to_jsonable_python(new_messages))
            _merge_usage(total_usage, observed_usage)
            completed_requests = int(observed_usage.get("requests", 0) or 0)
            safe_history = _safe_recovery_history(
                captured,
                prefix_count=len(history),
            )

            parse_failure = _tool_call_parse_failure(exc)
            boundary_failure = _generation_boundary_failure(
                max_tokens,
                to_jsonable_python(captured),
            )
            if parse_failure is not None:
                used_requests += completed_requests + 1
                total_usage["requests"] = int(total_usage.get("requests", 0) or 0) + 1
                history = safe_history
                next_prompt = _tool_call_parse_feedback(parse_failure)
                transient_failures = 0
                print(f"pm-coder recovery: {parse_failure}", file=sys.stderr, flush=True)
                continue
            if boundary_failure is not None:
                error_class, detail = boundary_failure
                if error_class is ContextLimitReachedError:
                    raise error_class(detail) from exc
                used_requests += completed_requests
                history = safe_history
                next_prompt = _response_limit_feedback(max_tokens)
                transient_failures = 0
                print(f"pm-coder recovery: {detail}", file=sys.stderr, flush=True)
                continue
            context_failure = _context_limit_failure(exc)
            if context_failure is not None:
                raise ContextLimitReachedError(context_failure) from exc
            if _is_transient_endpoint_failure(exc):
                used_requests += completed_requests
                history, next_prompt = _transient_recovery_state(
                    captured,
                    prefix_count=len(history),
                    retry_prompt=next_prompt,
                )
                transient_failures += 1
                delay = _transient_retry_delay(exc, transient_failures)
                print(
                    "pm-coder endpoint unavailable; retrying in "
                    f"{delay:g}s without charging the failed request",
                    file=sys.stderr,
                    flush=True,
                )
                remaining = _remaining_wall_clock(deadline)
                if remaining is not None and delay >= remaining:
                    raise TimeoutError(
                        "the model endpoint did not recover before the wall-clock limit"
                    ) from exc
                await asyncio.sleep(delay)
                continue
            endpoint_failure = _endpoint_failure(exc)
            if endpoint_failure is not None:
                error_class, detail = endpoint_failure
                raise error_class(detail) from exc
            raise

        result_usage = _usage_dict(result.usage)
        boundary_failure = _generation_boundary_failure(
            max_tokens,
            to_jsonable_python(result.all_messages()),
        )
        if boundary_failure is None:
            _merge_usage(total_usage, result_usage)
            return result, captured, total_usage

        error_class, detail = boundary_failure
        _merge_usage(total_usage, result_usage)
        used_requests += int(result_usage.get("requests", 0) or 0)
        if error_class is ContextLimitReachedError:
            raise error_class(detail)
        history = _safe_recovery_history(
            result.all_messages(),
            prefix_count=len(history),
        )
        next_prompt = _response_limit_feedback(max_tokens)
        transient_failures = 0
        print(f"pm-coder recovery: {detail}", file=sys.stderr, flush=True)


async def async_run_auto(
    prompt_or_path: str | Path,
    *,
    cwd: str | Path | None = None,
    run_id: str | None = None,
    log_root: str | Path = DEFAULT_LOG_ROOT,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    mcp_config: str | Path | None = None,
    shell: str = "auto",
    shell_timeout: int = DEFAULT_SHELL_TIMEOUT,
    max_tool_output: int | None = DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    max_skill_index: int | None = DEFAULT_MAX_SKILL_INDEX_CHARS,
    max_project_instructions: int | None = DEFAULT_MAX_PROJECT_INSTRUCTIONS_CHARS,
    temperature: float = 0.1,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    enable_thinking: bool = True,
    request_limit: int | None = None,
    wall_clock_limit: int | None = None,
) -> AutoResult:
    """Run one prompt and return a JSON-compatible result.

    ``prompt_or_path`` is interpreted as a UTF-8 text-file path when it names
    an existing file; otherwise it is used as literal prompt text. Pass the
    ``run_id`` returned by an earlier call to continue that exact session.
    Every successful turn updates ``messages.json`` before returning.
    """
    cwd_path = Path(cwd or os.getcwd()).expanduser().resolve()
    argv = ["--cwd", str(cwd_path), "--shell", shell, "--shell-timeout", str(shell_timeout)]
    if base_url is not None:
        argv += ["--base-url", base_url]
    if api_key is not None:
        argv += ["--api-key", api_key]
    if model is not None:
        argv += ["--model", model]
    if mcp_config is not None:
        argv += ["--mcp-config", str(mcp_config)]
    for option, value in (
        ("--max-tool-output", max_tool_output),
        ("--max-skill-index", max_skill_index),
        ("--max-project-instructions", max_project_instructions),
        ("--max-tokens", max_tokens),
    ):
        if value is not None:
            argv += [option, str(value)]
    argv += ["--temperature", str(temperature)]
    if enable_thinking:
        argv.append("--enable-thinking")
    settings = build_settings(parse_args(argv))
    discovery = discover_workspace(settings)
    session = SessionStore.open(
        settings.cwd,
        run_id,
        log_root=Path(log_root).expanduser(),
    )
    prompt = _prompt_text(prompt_or_path, cwd=settings.cwd)
    history = session.load_messages()
    agent = build_agent(settings, discovery)
    started = time.perf_counter()
    try:
        async with agent:
            result, captured, turn_usage = await _run_agent_turn(
                agent,
                prompt,
                history,
                max_tokens=settings.max_tokens,
                request_limit=request_limit,
                wall_clock_limit=wall_clock_limit,
            )
        messages = result.all_messages()
        session.save_messages(messages)
        duration = round(time.perf_counter() - started, 6)
        usage = turn_usage
        output = AutoResult(
            response=str(result.output),
            run_id=session.run_id,
            duration_seconds=duration,
            tokens_used=usage,
        )
        session.append_run(
            {
                "timestamp": utc_now(),
                "prompt": prompt,
                **output.as_dict(),
            }
        )
        return output
    except BaseException:
        if 'captured' in locals() and captured:
            session.save_messages(captured)
        raise


def run_auto(
    prompt_or_path: str | Path,
    *,
    cwd: str | Path | None = None,
    run_id: str | None = None,
    log_root: str | Path = DEFAULT_LOG_ROOT,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    mcp_config: str | Path | None = None,
    shell: str = "auto",
    shell_timeout: int = DEFAULT_SHELL_TIMEOUT,
    max_tool_output: int | None = DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    max_skill_index: int | None = DEFAULT_MAX_SKILL_INDEX_CHARS,
    max_project_instructions: int | None = DEFAULT_MAX_PROJECT_INSTRUCTIONS_CHARS,
    temperature: float = 0.1,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    enable_thinking: bool = True,
    request_limit: int | None = None,
    wall_clock_limit: int | None = None,
) -> dict[str, Any]:
    """Synchronous counterpart to :func:`async_run_auto`.

    This is the function-equivalent of auto CLI mode and is intended for
    simple loops such as ``run_auto(input())``. ``prompt_or_path`` is literal
    text unless it names an existing UTF-8 text file. Use ``run_id`` from the
    previous result to continue the same conversation. Use
    ``async_run_auto`` from an already-running asyncio event loop.
    """
    kwargs = {
        "cwd": cwd,
        "run_id": run_id,
        "log_root": log_root,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "mcp_config": mcp_config,
        "shell": shell,
        "shell_timeout": shell_timeout,
        "max_tool_output": max_tool_output,
        "max_skill_index": max_skill_index,
        "max_project_instructions": max_project_instructions,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "request_limit": request_limit,
        "wall_clock_limit": wall_clock_limit,
    }
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(async_run_auto(prompt_or_path, **kwargs)).as_dict()
    raise RuntimeError("run_auto cannot run inside an active event loop; use async_run_auto")


def run_auto_sync(
    prompt_or_path: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Explicitly named synchronous alias for :func:`run_auto`.

    It creates and owns a temporary asyncio loop internally, so callers can
    use it from ordinary synchronous code or a regular worker thread.
    """
    return run_auto(prompt_or_path, **kwargs)


def print_startup(settings: Settings, discovery: DiscoveryResult) -> None:
    print(f"\n{APP_NAME}")
    print(f"  cwd:        {settings.cwd}")
    print(f"  endpoint:   {settings.base_url}")
    print(f"  model:      {settings.model}")
    print(f"  shell:      {settings.shell_kind} ({settings.shell_executable})")
    print(f"  skills:     {len(discovery.skills)}")
    print(f"  instructions: {len(discovery.instruction_files)} file(s)")
    print(
        "  MCP servers: "
        + (", ".join(discovery.mcp_server_names) or "none configured")
    )
    for error in discovery.skill_errors:
        print(f"  skill warning: {error}")


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
    session: SessionStore,
) -> None:
    message_history = session.load_messages()
    print_startup(settings, discovery)
    print(f"  run id:      {session.run_id}")
    async with agent:
        while True:
            prompt = read_user_prompt()
            if prompt is None:
                return
            command = prompt.strip().casefold()
            if command in {"/quit", "/exit", "quit", "exit"}:
                return
            if command == "/clear":
                message_history = None
                session.clear()
                continue
            if command == "/info":
                print_startup(settings, discovery)
                continue
            if not prompt.strip():
                continue
            result, _captured, turn_usage = await _run_agent_turn(
                agent,
                prompt,
                message_history,
                max_tokens=settings.max_tokens,
            )
            message_history = result.all_messages()
            session.save_messages(message_history)
            session.append_run(
                {
                    "timestamp": utc_now(),
                    "prompt": prompt,
                    "response": str(result.output),
                    "tokens_used": turn_usage,
                }
            )
            print(f"\nAgent> {result.output}\n")


async def async_main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = build_settings(args)
    discovery = discover_workspace(settings)
    options = one_shot_options(args)
    if options is not None:
        os.chdir(settings.cwd)
        agent = build_agent(
            settings,
            discovery,
            plan_checkpoint=plan_checkpoint_for_one_shot(settings, options),
        )
        output = await run_one_shot(agent, settings, discovery, options)
        print(output)
        return
    if args.mode == "auto":
        prompt = args.prompt_file or args.prompt
        if prompt is None:
            raise ValueError("auto mode requires prompt text or a prompt file")
        result = await async_run_auto(
            prompt,
            cwd=settings.cwd,
            run_id=args.run_id,
            log_root=args.log_root,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            mcp_config=args.mcp_config,
            shell=args.shell,
            shell_timeout=args.shell_timeout,
            max_tool_output=args.max_tool_output,
            max_skill_index=args.max_skill_index,
            max_project_instructions=args.max_project_instructions,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
            request_limit=args.request_limit,
            wall_clock_limit=args.wall_clock_limit,
        )
        print(json.dumps(result.as_dict(), ensure_ascii=False))
        return
    if args.prompt is not None or args.prompt_file is not None:
        raise ValueError("a prompt requires --mode auto")
    session = SessionStore.open(
        settings.cwd,
        args.run_id,
        log_root=Path(args.log_root).expanduser(),
    )
    agent = build_agent(
        settings,
        discovery,
    )
    await interactive_loop(agent, settings, discovery, session)


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(
            f"{APP_NAME} failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
