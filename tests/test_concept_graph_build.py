from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "VectorDB") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "VectorDB"))

from graph_vocab import RELATION_TYPES, SUPPORT_LEVELS, all_entities
from scripts.build_concept_graph import build_concept_graph

from conftest import requires_hierarchical_artifacts

pytestmark = requires_hierarchical_artifacts


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_concept_graph_build_outputs_valid_deterministic_artifacts(tmp_path):
    out_dir = tmp_path / "graph"
    manifest1 = build_concept_graph(graph_dir=out_dir)
    nodes1 = read_jsonl(out_dir / "nodes.jsonl")
    edges1 = read_jsonl(out_dir / "edges.jsonl")

    required = {"nodes.jsonl", "edges.jsonl", "paths.jsonl", "claims.jsonl", "graph_manifest.json"}
    assert required <= {path.name for path in out_dir.iterdir()}
    assert manifest1["graph_counts"]["nodes"] > 0
    assert manifest1["graph_counts"]["edges"] > 0
    assert manifest1["graph_counts"]["claims"] == manifest1["graph_counts"]["edges"]

    manifest2 = build_concept_graph(graph_dir=out_dir)
    nodes2 = read_jsonl(out_dir / "nodes.jsonl")
    assert [node["node_id"] for node in nodes1] == [node["node_id"] for node in nodes2]
    assert manifest2["graph_counts"] == manifest1["graph_counts"]

    assert all(edge.get("source_span_ids") for edge in edges1)
    assert all(edge.get("relation_type") in RELATION_TYPES for edge in edges1)
    assert all(edge.get("support_level") in SUPPORT_LEVELS for edge in edges1)

    node_type_by_id = {data["node_id"]: data["node_type"] for data in all_entities().values()}
    for edge in edges1:
        target_type = node_type_by_id.get(edge.get("target_node_id"))
        if edge.get("relation_type") == "supplies":
            assert target_type not in {"symptom", "condition", "red_flag"}
        if edge.get("relation_type") == "innervates":
            assert target_type not in {"symptom", "condition", "red_flag"}

        source = edge.get("source_node_id")
        target = edge.get("target_node_id")
        if edge.get("relation_type") in {"compresses", "may_compress"}:
            assert (source, target) not in {
                ("node_clavicle", "node_lumbar_plexus"),
                ("node_thoracic_outlet", "node_lumbar_plexus"),
                ("node_scalene", "node_lumbar_plexus"),
                ("node_eustachian_tube", "node_vagus_nerve"),
                ("node_eustachian_tube", "node_phrenic_nerve"),
            }
        if edge.get("relation_type") == "stabilizes":
            assert (source, target) != ("node_rotator_cuff", "node_scapula")
