from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_graph_nerve_completeness import REQUIRED_CASE_FIELDS, evaluate, load_cases


DATASET = PROJECT_ROOT / "datasets" / "graph-nerve-completeness-cases.jsonl"
GRAPH_DIR = PROJECT_ROOT / "MSKArticlesINDEX" / "graph"
SCRIPT = PROJECT_ROOT / "scripts" / "eval_graph_nerve_completeness.py"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nerve_completeness_dataset_loads_and_has_required_fields():
    cases = load_cases(DATASET)
    assert len(cases) == 11
    assert {case["case_id"] for case in cases} == {
        "nerve_001_accessory_nerve",
        "nerve_002_dorsal_scapular_nerve",
        "nerve_003_trigeminal_nerve",
        "nerve_004_auriculotemporal_nerve",
        "nerve_005_vagus_nerve",
        "nerve_006_phrenic_nerve",
        "nerve_007_cervical_plexus",
        "nerve_008_occipital_nerves",
        "nerve_009_brachial_plexus",
        "nerve_010_lumbar_plexus",
        "nerve_011_pudendal_nerve",
    }
    for case in cases:
        assert set(REQUIRED_CASE_FIELDS) <= set(case)
        assert isinstance(case["required_supporting_span_count"], int)
        for field in REQUIRED_CASE_FIELDS:
            if field not in {"case_id", "query", "required_supporting_span_count", "notes"}:
                assert isinstance(case[field], list), f"{case['case_id']} field {field} must be a list"


def test_script_runs_and_writes_report_files(tmp_path):
    output_md = tmp_path / "report.md"
    output_json = tmp_path / "results.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert output_md.exists()
    assert output_json.exists()
    data = read_json(output_json)
    assert data["summary"]["total_cases"] == 11
    assert "Graph Nerve Completeness Report" in output_md.read_text(encoding="utf-8")


def test_missing_expected_nodes_are_reported_clearly(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    dataset = tmp_path / "cases.jsonl"
    write_jsonl(graph_dir / "nodes.jsonl", [])
    write_jsonl(graph_dir / "edges.jsonl", [])
    write_jsonl(graph_dir / "paths.jsonl", [])
    write_jsonl(graph_dir / "claims.jsonl", [])
    write_jsonl(
        dataset,
        [
            {
                "case_id": "missing_node_case",
                "query": "accessory nerve",
                "expected_nodes": ["accessory nerve"],
                "expected_aliases": [],
                "expected_muscles": [],
                "expected_compression_sites": [],
                "expected_symptoms": [],
                "expected_tests_or_assessments": [],
                "expected_exercise_or_posture_terms": [],
                "expected_article_families": [],
                "forbidden_false_positives": [],
                "required_supporting_span_count": 0,
                "notes": "fixture",
            }
        ],
    )

    results = evaluate(graph_dir, dataset)
    case = results["cases"][0]
    assert case["passed"] is False
    assert case["missing_by_category"]["node_coverage"] == ["accessory nerve"]


def test_weak_forbidden_co_mentions_are_classified_but_do_not_fail_case(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    dataset = tmp_path / "cases.jsonl"
    write_jsonl(
        graph_dir / "nodes.jsonl",
        [
            {
                "node_id": "node_accessory_nerve",
                "canonical_name": "accessory nerve",
                "aliases": ["spinal accessory nerve"],
                "source_span_ids": ["span1"],
                "source_article_ids": ["article1"],
            },
            {
                "node_id": "node_vagus_nerve",
                "canonical_name": "vagus nerve",
                "aliases": ["cranial nerve x"],
                "source_span_ids": ["span2"],
                "source_article_ids": ["article1"],
            },
        ],
    )
    write_jsonl(
        graph_dir / "edges.jsonl",
        [
            {
                "edge_id": "edge1",
                "source_node_id": "node_accessory_nerve",
                "target_node_id": "node_vagus_nerve",
                "relation_type": "mentioned_with",
                "support_level": "weak",
                "evidence_text": "accessory nerve and vagus nerve false fixture",
                "source_span_ids": ["span1"],
                "source_article_ids": ["article1"],
            }
        ],
    )
    write_jsonl(
        graph_dir / "paths.jsonl",
        [
            {
                "path_id": "path1",
                "node_ids": ["node_accessory_nerve"],
                "path_text": "accessory nerve",
                "weakest_support_level": "direct",
            }
        ],
    )
    write_jsonl(graph_dir / "claims.jsonl", [])
    write_jsonl(
        dataset,
        [
            {
                "case_id": "forbidden_case",
                "query": "accessory nerve",
                "expected_nodes": ["accessory nerve"],
                "expected_aliases": ["spinal accessory nerve"],
                "expected_muscles": [],
                "expected_compression_sites": [],
                "expected_symptoms": [],
                "expected_tests_or_assessments": [],
                "expected_exercise_or_posture_terms": [],
                "expected_article_families": [],
                "forbidden_false_positives": ["vagus nerve"],
                "required_supporting_span_count": 1,
                "notes": "fixture",
            }
        ],
    )

    results = evaluate(graph_dir, dataset)
    case = results["cases"][0]
    assert case["forbidden_false_positives"]["present"] == []
    assert case["forbidden_false_positives"]["weak_only_present"] == ["vagus nerve"]
    assert case["forbidden_false_positives"]["passed"] is True


def test_meaningful_forbidden_false_positives_fail_case(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    dataset = tmp_path / "cases.jsonl"
    write_jsonl(
        graph_dir / "nodes.jsonl",
        [
            {
                "node_id": "node_accessory_nerve",
                "canonical_name": "accessory nerve",
                "aliases": ["spinal accessory nerve"],
                "source_span_ids": ["span1"],
                "source_article_ids": ["article1"],
            },
            {
                "node_id": "node_vagus_nerve",
                "canonical_name": "vagus nerve",
                "aliases": ["cranial nerve x"],
                "source_span_ids": ["span2"],
                "source_article_ids": ["article1"],
            },
        ],
    )
    write_jsonl(
        graph_dir / "edges.jsonl",
        [
            {
                "edge_id": "edge1",
                "source_node_id": "node_accessory_nerve",
                "target_node_id": "node_vagus_nerve",
                "relation_type": "may_contribute_to",
                "support_level": "indirect",
                "evidence_text": "accessory nerve and vagus nerve fixture",
                "source_span_ids": ["span1"],
                "source_article_ids": ["article1"],
            }
        ],
    )
    write_jsonl(
        graph_dir / "paths.jsonl",
        [
            {
                "path_id": "path1",
                "node_ids": ["node_accessory_nerve", "node_vagus_nerve"],
                "path_text": "accessory nerve -> vagus nerve",
                "weakest_support_level": "indirect",
            }
        ],
    )
    write_jsonl(graph_dir / "claims.jsonl", [])
    write_jsonl(
        dataset,
        [
            {
                "case_id": "forbidden_case",
                "query": "accessory nerve",
                "expected_nodes": ["accessory nerve"],
                "expected_aliases": ["spinal accessory nerve"],
                "expected_muscles": [],
                "expected_compression_sites": [],
                "expected_symptoms": [],
                "expected_tests_or_assessments": [],
                "expected_exercise_or_posture_terms": [],
                "expected_article_families": [],
                "forbidden_false_positives": ["vagus nerve"],
                "required_supporting_span_count": 1,
                "notes": "fixture",
            }
        ],
    )

    results = evaluate(graph_dir, dataset)
    case = results["cases"][0]
    assert case["forbidden_false_positives"]["present"] == ["vagus nerve"]
    assert case["forbidden_false_positives"]["weak_only_present"] == []
    assert case["forbidden_false_positives"]["passed"] is False


def test_evaluation_does_not_mutate_graph_artifacts(tmp_path):
    graph_files = ["nodes.jsonl", "edges.jsonl", "paths.jsonl", "claims.jsonl"]
    before = {name: hash_file(GRAPH_DIR / name) for name in graph_files}
    output_md = tmp_path / "report.md"
    output_json = tmp_path / "results.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-md",
            str(output_md),
            "--output-json",
            str(output_json),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    after = {name: hash_file(GRAPH_DIR / name) for name in graph_files}
    assert after == before
