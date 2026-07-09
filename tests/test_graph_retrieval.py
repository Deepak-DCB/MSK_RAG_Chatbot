from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "VectorDB") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "VectorDB"))

import qaEngine
from graph_retrieval import build_graph_context, find_nodes, load_graph


def test_graph_retrieval_finds_tos_related_paths():
    pack = build_graph_context("How can scapular depression relate to thoracic outlet symptoms?")
    assert pack["available"] is True
    assert pack["paths"] or pack["edges"]
    path_text = " ".join(path.get("path_text", "") for path in pack["paths"]).lower()
    edge_text = " ".join(f"{edge.get('source')} {edge.get('target')}" for edge in pack["edges"]).lower()
    assert "scapular depression" in path_text or "scapular depression" in edge_text
    assert "thoracic outlet" in path_text or "thoracic outlet" in edge_text or "costoclavicular" in path_text


def test_graph_retrieval_finds_relevant_nodes():
    graph = load_graph()
    for query, expected in [
        ("scalenes", "scalene"),
        ("costoclavicular", "costoclavicular"),
        ("ulnar paresthesia", "ulnar"),
        ("cervical plexus", "cervical plexus"),
    ]:
        names = [node.get("canonical_name", "").lower() for node in find_nodes(query, graph)]
        assert any(expected in name for name in names)


def test_graph_retrieval_avoids_known_broad_alias_false_positives():
    graph = load_graph()

    anterior_tilt_names = [node.get("canonical_name", "") for node in find_nodes("anterior pelvic tilt and lumbar lordosis", graph)]
    assert "anterior pelvic tilt" in anterior_tilt_names
    assert "anterior scapular tilt" not in anterior_tilt_names

    plexus_names = [node.get("canonical_name", "") for node in find_nodes("lumbar plexus and pudendal neuralgia", graph)]
    assert "lumbar plexus" in plexus_names
    assert "brachial plexus" not in plexus_names

    axis_names = [node.get("canonical_name", "") for node in find_nodes("humeral axis and axis of rotation", graph)]
    assert "axis" not in axis_names

    levator_names = [node.get("canonical_name", "") for node in find_nodes("levator veli palatini and eustachian tube", graph)]
    assert "levator scapulae" not in levator_names


def test_graph_retrieval_handles_missing_artifacts(tmp_path):
    pack = build_graph_context("thoracic outlet", base_dir=tmp_path / "missing")
    assert pack["available"] is False
    assert pack["fallback_reason"]


def test_graph_context_token_estimate_respects_max_tokens():
    pack = build_graph_context("scapular depression thoracic outlet paresthesia", max_graph_tokens=80)
    assert pack["context_token_estimate"] <= 80


def test_qaengine_falls_back_when_graph_artifacts_unavailable(monkeypatch):
    def fake_graph_context(*args, **kwargs):
        return {
            "available": False,
            "fallback_reason": "graph_artifacts_missing",
            "nodes": [],
            "edges": [],
            "paths": [],
            "supporting_spans": [],
            "context_token_estimate": 0,
            "context": "",
        }

    monkeypatch.setattr(qaEngine, "build_graph_context", fake_graph_context)
    context = [
        {
            "text": "Scapular depression may affect thoracic outlet mechanics.",
            "meta": {"source_relpath": "source", "section": "Main", "chunk_id": "c1"},
            "dist": 0.1,
        }
    ]
    pack = qaEngine.build_context_pack(context, "scapular depression and thoracic outlet", qaEngine.QAConfig(context_strategy="chunk_pack"))
    assert pack.graph_available is False
    assert pack.graph_fallback_reason == "graph_artifacts_missing"
    assert pack.formatted_context
