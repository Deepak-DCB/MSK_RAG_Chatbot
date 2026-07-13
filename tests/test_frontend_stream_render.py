"""Runs the Node behavioural tests for the frontend streaming render path.

The logic under test is a DOM/animation-frame race, so it cannot be meaningfully
asserted from Python by string-matching app.js. tests/frontend/stream_render.test.mjs
loads the real frontend/app.js in a vm with a fake DOM and a MANUALLY PUMPED
requestAnimationFrame, which lets it force the exact hostile interleaving:
a frame queued by the final token firing AFTER stream finalization has already
appended the incomplete-stream warning, citations, telemetry and feedback.

Skips (does not fail) when node is unavailable, so the Python-only environment
still runs the rest of the suite.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NODE_TEST = PROJECT_ROOT / "tests" / "frontend" / "stream_render.test.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_frontend_streaming_render_race():
    assert NODE_TEST.exists(), f"missing node test: {NODE_TEST}"

    proc = subprocess.run(
        ["node", str(NODE_TEST)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )

    if proc.returncode != 0:
        pytest.fail(
            "frontend stream-render tests failed:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
