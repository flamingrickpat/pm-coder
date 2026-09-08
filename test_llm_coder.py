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


def _solid_png(path, rgb, size=64):
    """A real PNG file, built by hand -- no image library in the test env."""
    import struct
    import zlib

    def chunk(tag, data):
        raw = tag + data
        return (
            struct.pack(">I", len(data))
            + raw
            + struct.pack(">I", zlib.crc32(raw))
        )

    row = b"\x00" + bytes(rgb) * size
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


def test_file_tools_roundtrip_on_real_filesystem(tmp_path):
    """The agent uses write, edit, and read on a real scratch directory."""
    base = _require_endpoint()
    result = run_auto(
        "Do exactly these three steps, in order, using the file tools:\n"
        "1. Use write to create hello.txt containing exactly:\n"
        "alpha\n"
        "beta\n"
        "gamma\n"
        "2. Use edit to replace the line 'beta' with 'BETA' in hello.txt.\n"
        "3. Use read to read hello.txt back, then reply with its content.",
        cwd=tmp_path,
        base_url=base,
    )
    lines = (tmp_path / "hello.txt").read_text(encoding="utf-8").splitlines()
    assert lines == ["alpha", "BETA", "gamma"], lines
    assert "BETA" in result["response"]


def test_read_attaches_image_the_model_can_see(tmp_path):
    """read() on a PNG attaches it; the mmproj endpoint must see the color."""
    base = _require_endpoint()
    _solid_png(tmp_path / "swatch.png", (220, 20, 20))
    result = run_auto(
        "Use the read tool on swatch.png, look at the attached image, and "
        "answer with the single dominant color as one word.",
        cwd=tmp_path,
        base_url=base,
    )
    assert "red" in result["response"].lower(), result["response"]


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