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
