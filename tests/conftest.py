"""Shared test helpers.

Some tests exercise artifacts that are built locally from the full corpus
mirror (gitignored — "do not publish") and are therefore absent in a clean
checkout and in CI: MSKArticlesINDEX/graph/, MSKArticlesINDEX/hierarchical/,
and MSKArticlesINDEX/mechanics/. Those tests skip cleanly when the artifacts
are missing (same convention as the in-test skip in test_review_regressions.py)
instead of failing on every fresh clone. Fixture-based tests in the same files
run everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_INDEX = PROJECT_ROOT / "MSKArticlesINDEX"

requires_graph_artifacts = pytest.mark.skipif(
    not (_INDEX / "graph" / "nodes.jsonl").exists(),
    reason="locally built concept-graph artifacts (MSKArticlesINDEX/graph/) not present",
)

requires_hierarchical_artifacts = pytest.mark.skipif(
    not (_INDEX / "hierarchical" / "evidence_spans.jsonl").exists(),
    reason="locally built hierarchical corpus (MSKArticlesINDEX/hierarchical/) not present",
)

requires_mechanics_artifacts = pytest.mark.skipif(
    not (_INDEX / "mechanics").is_dir(),
    reason="locally built mechanics maps (MSKArticlesINDEX/mechanics/) not present",
)
