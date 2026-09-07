from __future__ import annotations

import os
import re
import shutil
import signal
import sys
import time
from pathlib import Path

import pytest

from pm_coder import PowerShellBackend, ShellBackend, _run_host_shell


class PythonBackend(ShellBackend):
    kind = "bash"
    file_suffix = ".py"

    @property
    def preamble(self) -> str:
        return ""

    def invocation(self, script_path: str) -> list[str]:
        return [self.executable, script_path]


def test_shell_timeout_is_a_hard_wall_clock(tmp_path: Path) -> None:
    started = time.monotonic()

    result = _run_host_shell(
        PythonBackend(sys.executable),
        tmp_path,
        "import time\ntime.sleep(30)\n",
        timeout_seconds=1,
    )

    assert time.monotonic() - started < 4
    assert result.startswith("timed_out: true\ntimeout_seconds: 1\n")


def test_shell_timeout_cannot_be_disabled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be greater than zero"):
        _run_host_shell(
            PythonBackend(sys.executable),
            tmp_path,
            "print('never started')\n",
            timeout_seconds=0,
        )


def test_background_descendant_cannot_hold_tool_open(tmp_path: Path) -> None:
    started = time.monotonic()
    child_pid: int | None = None
    command = (
        "import subprocess, sys\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(30)'],\n"
        ")\n"
        "print(child.pid, flush=True)\n"
    )

    try:
        result = _run_host_shell(
            PythonBackend(sys.executable),
            tmp_path,
            command,
            timeout_seconds=1,
        )
        match = re.search(r"stdout:\n(\d+)", result)
        assert match is not None
        child_pid = int(match.group(1))

        assert time.monotonic() - started < 4
        assert result.startswith("exit_code: 0\n")
        assert child_pid > 0
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell regression")
def test_powershell_start_process_cannot_hold_tool_open(tmp_path: Path) -> None:
    powershell = shutil.which("powershell.exe")
    assert powershell is not None
    started = time.monotonic()
    child_pid: int | None = None
    command = (
        '$child = Start-Process -FilePath "$env:SystemRoot\\System32\\ping.exe" '
        '-ArgumentList "-t","127.0.0.1" -PassThru\n'
        "Write-Output $child.Id\n"
    )

    try:
        result = _run_host_shell(
            PowerShellBackend(powershell),
            tmp_path,
            command,
            timeout_seconds=1,
        )
        match = re.search(r"stdout:\n\s*(\d+)", result)
        assert match is not None
        child_pid = int(match.group(1))

        assert time.monotonic() - started < 4
        assert result.startswith("exit_code: 0\n")
        assert child_pid > 0
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def test_large_shell_output_has_bounded_tail_and_complete_logs(tmp_path):
    from pm_coder import context_limits

    logs = tmp_path / "logs"
    result = _run_host_shell(
        PythonBackend(sys.executable), tmp_path,
        "import sys\nfor i in range(1000): print(f'row-{i:04d}')\n"
        "print('E' * 20000, file=sys.stderr)\nsys.exit(7)\n",
        timeout_seconds=5, log_dir=logs,
    )
    assert result.startswith("exit_code: 7\n")
    assert "output truncated" in result
    assert "row-0000" not in result
    assert "row-0900" in result and "row-0999" in result
    assert len(result) < 2 * context_limits().shell_chars + 1500
    stdout_log = next(logs.glob("*/stdout.log"))
    stderr_log = next(logs.glob("*/stderr.log"))
    assert stdout_log.read_text().splitlines() == [f"row-{i:04d}" for i in range(1000)]
    assert stderr_log.read_text().strip() == "E" * 20000
    assert str(stdout_log) in result and str(stderr_log) in result


def test_timeout_retains_partial_output_in_logs(tmp_path):
    logs = tmp_path / "logs"
    result = _run_host_shell(
        PythonBackend(sys.executable), tmp_path,
        "import sys, time\nprint('before timeout', flush=True)\n"
        "print('diagnostic', file=sys.stderr, flush=True)\ntime.sleep(30)\n",
        timeout_seconds=1, log_dir=logs,
    )
    assert result.startswith("timed_out: true")
    assert "before timeout" in result and "diagnostic" in result
    assert next(logs.glob("*/stdout.log")).read_text().strip() == "before timeout"
    assert next(logs.glob("*/stderr.log")).read_text().strip() == "diagnostic"


def test_shell_calls_keep_separate_logs(tmp_path):
    logs = tmp_path / "logs"
    for word in ("first", "second"):
        result = _run_host_shell(
            PythonBackend(sys.executable), tmp_path, f"print('{word}')",
            timeout_seconds=5, log_dir=logs,
        )
        assert f"stdout:\n{word}" in result
        assert "output truncated" not in result
    assert sorted(p.read_text().strip() for p in logs.glob("*/stdout.log")) == ["first", "second"]


def setup_module():
    import tempfile
    from pathlib import Path
    import pm_coder

    root = Path(tempfile.mkdtemp(prefix="pm-coder-check-"))
    pm_coder.active_session = pm_coder.SessionStore.open(root, log_root=root / "sessions")
    pm_coder.active_session.context_window = 96_000
