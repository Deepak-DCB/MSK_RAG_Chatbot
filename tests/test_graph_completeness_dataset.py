from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "VectorDB") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "VectorDB"))

from graph_retrieval import build_graph_context


def read_completeness_cases():
    path = PROJECT_ROOT / "datasets" / "graph-completeness-cases.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_completeness_dataset_covers_all_20_articles_and_gap_status():
    cases = read_completeness_cases()
    assert len(cases) == 20
    assert all("known_gap" in case for case in cases)
    assert len({case.get("article") for case in cases}) == 20


@pytest.mark.parametrize("case", read_completeness_cases(), ids=lambda c: c["case_id"])
def test_full_article_completeness_cases(case):
    if case.get("known_gap"):
        pytest.xfail(f"Known graph completeness gap: {case['current_status']}")

    pack = build_graph_context(case["question"], max_graph_tokens=1000)
    assert pack["available"] is True
    assert pack.get("fallback_reason") is None

    node_names = {node.get("canonical_name") for node in pack.get("nodes", [])}
    missing = [name for name in case.get("expected_nodes", []) if name not in node_names]
    assert not missing, f"missing nodes: {missing}; got {sorted(node_names)}"

    forbidden = [name for name in case.get("forbidden_nodes", []) if name in node_names]
    assert not forbidden, f"forbidden nodes present: {forbidden}"

    path_text = "\n".join(path.get("path_text", "") for path in pack.get("paths", [])).lower()
    for term in case.get("expected_path_terms", []):
        assert term.lower() in path_text, f"missing path term {term!r}; paths={path_text!r}"

    expected_policy = case.get("expected_policy")
    if expected_policy:
        policies = {path.get("clinical_policy") for path in pack.get("paths", [])}
        assert expected_policy in policies
