"""Live-LLM tests. They fail unless an OpenAI-compatible endpoint answers.

Set LOCAL_AGENT_BASE_URL (or OPENAI_BASE_URL) to point elsewhere.
Default: http://127.0.0.1:8080/v1

Run:
    pytest -v test_llm_coder.py
"""

from __future__ import annotations

import json
import os
import urllib.request

import pytest

from pm_bash_machine import BashMachine
from pm_coder import run_auto, run_auto_with_bash_machine

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


def _require_endpoint() -> str:
    base = (
        os.environ.get("LOCAL_AGENT_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    ).rstrip("/")
    try:
        with urllib.request.urlopen(base + "/models", timeout=5) as response:
            json.loads(response.read())
    except Exception as exc:
        pytest.fail(f"live LLM required at {base}: {exc}")
    return base


def test_tetris_on_real_filesystem(tmp_path):
    """Create a tetris.html on a real temp directory, then delete it."""
    base = _require_endpoint()
    run_auto(
        "Create a complete, playable Tetris game in a single HTML file "
        "named tetris.html. Use HTML, CSS, and JavaScript all in that "
        "one file. The game must render a board, spawn pieces, and "
        "accept keyboard controls. Write the full file.",
        cwd=tmp_path,
        base_url=base,
    )
    html = tmp_path / "tetris.html"
    assert html.is_file(), "tetris.html was not created"
    content = html.read_text(encoding="utf-8")
    assert content.strip(), "tetris.html is empty"
    html.unlink()


def test_tetris_in_bash_machine(tmp_path):
    """Create a tetris.html inside an in-memory BashMachine."""
    base = _require_endpoint()
    vm = BashMachine()
    run_auto_with_bash_machine(
        "Create a complete, playable Tetris game in a single HTML file "
        "at /home/user/tetris.html. Use HTML, CSS, and JavaScript all "
        "in that one file. The game must render a board, spawn pieces, "
        "and accept keyboard controls. Write the full file.",
        vm,
        cwd=tmp_path,
        base_url=base,
    )
    content = vm.read_text("/home/user/tetris.html")
    assert content.strip(), "tetris.html is empty or missing"