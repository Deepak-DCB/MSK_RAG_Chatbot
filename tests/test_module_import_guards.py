"""Import guards for first-party optional modules in qaEngine.

qaEngine degrades gracefully when hierarchical_retrieval / graph_retrieval are
absent — but only for a genuine ImportError. A real bug inside either module
(SyntaxError, NameError, ValueError at import time) must fail the qaEngine
import loudly instead of masquerading as "feature not installed", which is how
two graph incidents (b548b6f, 16511d8) stayed invisible until a deep review.

Subprocess-based: reloading qaEngine in-process is unsafe (module-level
singletons like _backend are shared with other test modules).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORDB_DIR = PROJECT_ROOT / "VectorDB"


def _import_qaengine_with_stub(tmp_path: Path, stub_source: str) -> subprocess.CompletedProcess:
    (tmp_path / "graph_retrieval.py").write_text(stub_source, encoding="utf-8")
    env = dict(os.environ)
    # Stub dir shadows the real VectorDB/graph_retrieval.py.
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(VECTORDB_DIR)])
    return subprocess.run(
        [sys.executable, "-c", "import qaEngine; print('IMPORT_OK')"],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_bug_in_first_party_module_fails_import_loudly(tmp_path):
    completed = _import_qaengine_with_stub(
        tmp_path, "raise ValueError('injected bug in graph_retrieval')\n"
    )
    assert completed.returncode != 0, (
        "a non-ImportError from graph_retrieval must not be swallowed:\n"
        + completed.stdout
    )
    assert "injected bug in graph_retrieval" in completed.stderr


def test_missing_dependency_still_degrades_gracefully(tmp_path):
    completed = _import_qaengine_with_stub(
        tmp_path, "raise ImportError('transitive dep not installed')\n"
    )
    assert completed.returncode == 0, completed.stderr
    assert "IMPORT_OK" in completed.stdout
    assert "graph_retrieval unavailable" in completed.stderr
