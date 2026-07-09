#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "MSKArticlesINDEX" / "graph"
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "graph-nerve-completeness-cases.jsonl"
DEFAULT_REPORT_MD = PROJECT_ROOT / "Evaluation" / "graph_nerve_completeness_report.md"
DEFAULT_RESULTS_JSON = PROJECT_ROOT / "Evaluation" / "graph_nerve_completeness_results.json"

REQUIRED_CASE_FIELDS = [
    "case_id",
    "query",
    "expected_nodes",
    "expected_aliases",
    "expected_muscles",
    "expected_compression_sites",
    "expected_symptoms",
    "expected_tests_or_assessments",
    "expected_exercise_or_posture_terms",
    "expected_article_families",
    "forbidden_false_positives",
    "required_supporting_span_count",
    "notes",
]

CATEGORY_FIELDS = {
    "expected_nodes": "node_coverage",
    "expected_aliases": "alias_coverage",
    "expected_muscles": "muscle_relationship_coverage",
    "expected_compression_sites": "compression_site_coverage",
    "expected_symptoms": "symptom_coverage",
    "expected_tests_or_assessments": "test_assessment_coverage",
    "expected_exercise_or_posture_terms": "exercise_posture_coverage",
}

MEANINGFUL_SUPPORT_LEVELS = {"direct", "indirect"}
WEAK_SUPPORT_LEVELS = {"weak", "inferred_from_same_section", "inferred_from_path", "unsupported"}
NON_MECHANISM_RELATIONS = {"mentioned_with"}


def edge_is_meaningful(edge: Mapping[str, Any]) -> bool:
    return (
        str(edge.get("support_level") or "weak") in MEANINGFUL_SUPPORT_LEVELS
        and str(edge.get("relation_type") or "") not in NON_MECHANISM_RELATIONS
    )


def path_is_meaningful(path: Mapping[str, Any]) -> bool:
    return str(path.get("weakest_support_level") or "weak") in MEANINGFUL_SUPPORT_LEVELS


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("_", " ").split())


def node_id_for_name(name: str) -> str:
    return "node_" + normalize(name).replace("'", "").replace("/", " ").replace("-", " ").replace(" ", "_")


def validate_case(case: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_CASE_FIELDS if field not in case]
    if missing:
        raise ValueError(f"{case.get('case_id', '<unknown>')} missing required fields: {missing}")
    for field in REQUIRED_CASE_FIELDS:
        if field in {"case_id", "query", "notes"}:
            if not isinstance(case[field], str):
                raise ValueError(f"{case.get('case_id')} field {field} must be a string")
        elif field == "required_supporting_span_count":
            if not isinstance(case[field], int) or case[field] < 0:
                raise ValueError(f"{case.get('case_id')} field {field} must be a non-negative integer")
        else:
            if not isinstance(case[field], list):
                raise ValueError(f"{case.get('case_id')} field {field} must be a list")


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases = read_jsonl(path)
    for case in cases:
        validate_case(case)
    return cases


def load_graph(graph_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "nodes": read_jsonl(graph_dir / "nodes.jsonl"),
        "edges": read_jsonl(graph_dir / "edges.jsonl"),
        "paths": read_jsonl(graph_dir / "paths.jsonl"),
        "claims": read_jsonl(graph_dir / "claims.jsonl"),
    }


def collect_case_evidence(case: Mapping[str, Any], graph: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    expected_node_terms = [str(term) for term in case.get("expected_nodes", [])]
    expected_alias_terms = [str(term) for term in case.get("expected_aliases", [])]
    relationship_terms = []
    for field in CATEGORY_FIELDS:
        if field not in {"expected_nodes", "expected_aliases"}:
            relationship_terms.extend(str(term) for term in case.get(field, []))
    query_terms = [term for term in [*expected_node_terms, *expected_alias_terms, *relationship_terms] if term]
    id_to_name = {str(node.get("node_id")): str(node.get("canonical_name", "")) for node in graph["nodes"]}
    id_to_aliases = {
        str(node.get("node_id")): [str(alias) for alias in node.get("aliases") or []]
        for node in graph["nodes"]
    }

    node_names: Set[str] = set()
    aliases: Set[str] = set()
    anchor_node_ids: Set[str] = set()
    primary_anchor_node_ids: Set[str] = set()
    evidence_text_parts: List[str] = []
    meaningful_text_parts: List[str] = []
    weak_text_parts: List[str] = []
    meaningful_names: Set[str] = set()
    weak_names: Set[str] = set()
    primary_meaningful_names: Set[str] = set()
    primary_weak_names: Set[str] = set()
    source_spans: Set[str] = set()
    source_articles: Set[str] = set()
    related_edges: List[Dict[str, Any]] = []
    related_paths: List[Dict[str, Any]] = []

    for node in graph["nodes"]:
        node_name = str(node.get("canonical_name", ""))
        node_aliases = [str(alias) for alias in node.get("aliases") or []]
        normalized_aliases = {normalize(value) for value in node_aliases}
        node_matches = any(normalize(term) == normalize(node_name) for term in expected_node_terms)
        alias_matches = any(normalize(term) in normalized_aliases for term in expected_alias_terms)
        if node_matches or alias_matches:
            node_names.add(node_name)
            aliases.update(node_aliases)
            node_id = str(node.get("node_id"))
            anchor_node_ids.add(node_id)
            if expected_node_terms and normalize(expected_node_terms[0]) == normalize(node_name):
                primary_anchor_node_ids.add(node_id)
            source_spans.update(str(span) for span in node.get("source_span_ids") or [])
            source_articles.update(str(article) for article in node.get("source_article_ids") or [])

    if not primary_anchor_node_ids:
        primary_anchor_node_ids = set(anchor_node_ids)

    for edge in graph["edges"]:
        edge_node_ids = {str(edge.get("source_node_id")), str(edge.get("target_node_id"))}
        endpoint_names = [id_to_name.get(node_id, node_id) for node_id in edge_node_ids]
        edge_text = " ".join(
            [*endpoint_names]
            + [
                str(edge.get(field, ""))
                for field in ["source_node_id", "target_node_id", "relation_type", "evidence_text", "notes"]
            ]
        )
        if edge_node_ids & anchor_node_ids:
            touches_primary = bool(edge_node_ids & primary_anchor_node_ids)
            related_edges.append(edge)
            evidence_text_parts.append(edge_text)
            if edge_is_meaningful(edge):
                for node_id in edge_node_ids:
                    node_name = id_to_name.get(node_id, node_id)
                    meaningful_names.add(node_name)
                    if touches_primary:
                        primary_meaningful_names.add(node_name)
                    meaningful_text_parts.append(node_name)
                    meaningful_text_parts.extend(id_to_aliases.get(node_id, []))
            else:
                for node_id in edge_node_ids:
                    node_name = id_to_name.get(node_id, node_id)
                    weak_names.add(node_name)
                    if touches_primary:
                        primary_weak_names.add(node_name)
                    weak_text_parts.append(node_name)
                    weak_text_parts.extend(id_to_aliases.get(node_id, []))
            source_spans.update(str(span) for span in edge.get("source_span_ids") or [])
            source_articles.update(str(article) for article in edge.get("source_article_ids") or [])

    for claim in graph["claims"]:
        claim_text = " ".join(str(claim.get(field, "")) for field in ["claim_text", "claim_type"])
        if set(str(node_id) for node_id in claim.get("involved_node_ids") or []) & anchor_node_ids:
            evidence_text_parts.append(claim_text)
            source_spans.update(str(span) for span in claim.get("source_span_ids") or [])
            source_articles.update(str(article) for article in claim.get("source_article_ids") or [])

    for path in graph["paths"]:
        path_node_names = [id_to_name.get(str(node_id), str(node_id)) for node_id in path.get("node_ids") or []]
        path_text = " ".join(
            [*path_node_names]
            + [str(path.get(field, "")) for field in ["path_family", "path_text", "clinical_policy"]]
        )
        if set(str(node_id) for node_id in path.get("node_ids") or []) & anchor_node_ids:
            path_node_ids = {str(node_id) for node_id in path.get("node_ids") or []}
            touches_primary = bool(path_node_ids & primary_anchor_node_ids)
            related_paths.append(path)
            evidence_text_parts.append(path_text)
            target_parts = meaningful_text_parts if path_is_meaningful(path) else weak_text_parts
            target_parts.append(path_text)
            for node_id in path.get("node_ids") or []:
                node_id = str(node_id)
                node_name = id_to_name.get(node_id, node_id)
                if path_is_meaningful(path):
                    meaningful_names.add(node_name)
                    if touches_primary:
                        primary_meaningful_names.add(node_name)
                else:
                    weak_names.add(node_name)
                    if touches_primary:
                        primary_weak_names.add(node_name)
                target_parts.append(node_name)
                target_parts.extend(id_to_aliases.get(node_id, []))
            source_spans.update(str(span) for span in path.get("source_span_ids") or [])
            source_articles.update(str(article) for article in path.get("source_article_ids") or [])

    searchable_terms = set(normalize(name) for name in node_names)
    searchable_terms.update(normalize(alias) for alias in aliases)
    searchable_terms.update(normalize(part) for part in evidence_text_parts if part)
    meaningful_terms = {normalize(part) for part in meaningful_text_parts if part}
    weak_terms = {normalize(part) for part in weak_text_parts if part}

    return {
        "node_names": node_names,
        "aliases": aliases,
        "searchable_text": "\n".join(sorted(searchable_terms)),
        "meaningful_text": "\n".join(sorted(meaningful_terms)),
        "weak_text": "\n".join(sorted(weak_terms)),
        "meaningful_names": meaningful_names,
        "weak_names": weak_names,
        "primary_meaningful_names": primary_meaningful_names,
        "primary_weak_names": primary_weak_names,
        "source_span_ids": sorted(source_spans),
        "source_article_ids": sorted(source_articles),
        "related_edges": related_edges,
        "related_paths": related_paths,
    }


def term_present(term: str, evidence: Mapping[str, Any], *, mode: str = "text") -> bool:
    normalized = normalize(term)
    if not normalized:
        return True
    exact_names = {normalize(name) for name in evidence["node_names"]}
    exact_aliases = {normalize(alias) for alias in evidence["aliases"]}
    if mode == "node":
        return normalized in exact_names
    if mode == "alias":
        return normalized in exact_aliases
    if mode == "meaningful":
        return normalized in exact_names or normalized in exact_aliases or normalized in str(evidence["meaningful_text"])
    if mode == "weak":
        return normalized in str(evidence["weak_text"])
    if mode == "forbidden_meaningful":
        return normalized in {normalize(name) for name in evidence["primary_meaningful_names"]}
    if mode == "forbidden_weak":
        return normalized in {normalize(name) for name in evidence["primary_weak_names"]}
    return normalized in exact_names or normalized in exact_aliases or normalized in str(evidence["searchable_text"])


def evaluate_terms(terms: Sequence[str], evidence: Mapping[str, Any], *, mode: str = "text") -> Dict[str, Any]:
    present = [term for term in terms if term_present(term, evidence, mode=mode)]
    weak_only = []
    if mode == "meaningful":
        weak_only = [term for term in terms if term not in present and term_present(term, evidence, mode="weak")]
    missing = [term for term in terms if term not in present]
    return {"expected": list(terms), "present": present, "weak_only": weak_only, "missing": missing, "passed": not missing}


def evaluate_case(case: Mapping[str, Any], graph: Mapping[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    evidence = collect_case_evidence(case, graph)
    category_results = {}
    for field, output_name in CATEGORY_FIELDS.items():
        mode = "node" if field == "expected_nodes" else "alias" if field == "expected_aliases" else "meaningful"
        category_results[output_name] = evaluate_terms(case.get(field, []), evidence, mode=mode)
    forbidden_strong_present = [term for term in case.get("forbidden_false_positives", []) if term_present(term, evidence, mode="forbidden_meaningful")]
    forbidden_weak_present = [
        term
        for term in case.get("forbidden_false_positives", [])
        if term not in forbidden_strong_present and term_present(term, evidence, mode="forbidden_weak")
    ]
    supporting_span_count = len(evidence["source_span_ids"])
    required_span_count = int(case.get("required_supporting_span_count", 0))
    paths = evidence["related_paths"]
    meaningful_paths = [path for path in paths if path_is_meaningful(path)]
    weak_paths = [path for path in paths if not path_is_meaningful(path)]

    missing_by_category = {
        name: result["missing"] for name, result in category_results.items() if result["missing"]
    }
    weak_only_by_category = {
        name: result["weak_only"] for name, result in category_results.items() if result.get("weak_only")
    }
    passed = (
        not missing_by_category
        and not forbidden_strong_present
        and supporting_span_count >= required_span_count
        and bool(meaningful_paths)
    )

    return {
        "case_id": case["case_id"],
        "query": case["query"],
        "notes": case.get("notes", ""),
        "passed": passed,
        "category_results": category_results,
        "forbidden_false_positives": {
            "expected_absent": list(case.get("forbidden_false_positives", [])),
            "present": forbidden_strong_present,
            "weak_only_present": forbidden_weak_present,
            "passed": not forbidden_strong_present,
        },
        "supporting_evidence": {
            "required_span_count": required_span_count,
            "actual_span_count": supporting_span_count,
            "source_span_ids": evidence["source_span_ids"],
            "source_article_ids": evidence["source_article_ids"],
            "passed": supporting_span_count >= required_span_count,
        },
        "path_presence": {
            "passed": bool(meaningful_paths),
            "path_count": len(paths),
            "meaningful_path_count": len(meaningful_paths),
            "weak_path_count": len(weak_paths),
            "paths": [
                {
                    "path_id": path.get("path_id"),
                    "path_family": path.get("path_family"),
                    "path_text": path.get("path_text"),
                    "clinical_policy": path.get("clinical_policy"),
                }
                for path in paths
            ],
        },
        "missing_by_category": missing_by_category,
        "weak_only_by_category": weak_only_by_category,
    }


def summarize(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    missing_nodes = Counter()
    missing_relationships = Counter()
    failed_categories = Counter()
    for result in results:
        for category, missing in result.get("missing_by_category", {}).items():
            failed_categories[category] += 1
            target = missing_nodes if category == "node_coverage" else missing_relationships
            target.update(missing)
        if not result.get("path_presence", {}).get("passed"):
            missing_relationships.update(["path presence"])
    return {
        "total_cases": len(results),
        "passed_cases": sum(1 for result in results if result.get("passed")),
        "failed_cases": sum(1 for result in results if not result.get("passed")),
        "top_missing_nodes": missing_nodes.most_common(20),
        "top_missing_relationships": missing_relationships.most_common(20),
        "failed_categories": failed_categories.most_common(),
    }


def render_markdown(results_doc: Mapping[str, Any]) -> str:
    summary = results_doc["summary"]
    lines = [
        "# Graph Nerve Completeness Report",
        "",
        f"Generated: {results_doc['generated_at']}",
        f"Dataset: `{results_doc['dataset']}`",
        f"Graph directory: `{results_doc['graph_dir']}`",
        "",
        "## Summary",
        "",
        f"- Cases: {summary['total_cases']}",
        f"- Passed: {summary['passed_cases']}",
        f"- Failed: {summary['failed_cases']}",
        "",
        "## Pass/Fail By Nerve",
        "",
        "| Case | Status | Missing Nodes | Missing Relationships | Forbidden Strong | Forbidden Weak-Only | Meaningful Paths | Total Paths | Spans |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    relationship_categories = [
        "alias_coverage",
        "muscle_relationship_coverage",
        "compression_site_coverage",
        "symptom_coverage",
        "test_assessment_coverage",
        "exercise_posture_coverage",
    ]
    for result in results_doc["cases"]:
        missing_nodes = ", ".join(result["missing_by_category"].get("node_coverage", [])) or "-"
        missing_relationships = []
        for category in relationship_categories:
            missing_relationships.extend(result["missing_by_category"].get(category, []))
        missing_relationships_text = ", ".join(missing_relationships) or "-"
        forbidden = ", ".join(result["forbidden_false_positives"]["present"]) or "-"
        forbidden_weak = ", ".join(result["forbidden_false_positives"].get("weak_only_present", [])) or "-"
        status = "PASS" if result["passed"] else "FAIL"
        spans = result["supporting_evidence"]["actual_span_count"]
        paths = result["path_presence"]["path_count"]
        meaningful_paths = result["path_presence"].get("meaningful_path_count", 0)
        lines.append(
            f"| {result['case_id']} | {status} | {missing_nodes} | {missing_relationships_text} | {forbidden} | {forbidden_weak} | {meaningful_paths} | {paths} | {spans} |"
        )

    lines.extend(["", "## Top Missing Nodes", ""])
    if summary["top_missing_nodes"]:
        lines.extend(f"- {term}: {count}" for term, count in summary["top_missing_nodes"])
    else:
        lines.append("- None")

    lines.extend(["", "## Top Missing Relationships", ""])
    if summary["top_missing_relationships"]:
        lines.extend(f"- {term}: {count}" for term, count in summary["top_missing_relationships"])
    else:
        lines.append("- None")

    lines.extend(["", "## Detailed Case Findings", ""])
    for result in results_doc["cases"]:
        lines.extend([
            f"### {result['case_id']}",
            "",
            f"Status: {'PASS' if result['passed'] else 'FAIL'}",
            f"Notes: {result['notes']}",
            "",
        ])
        for category, category_result in result["category_results"].items():
            missing = ", ".join(category_result["missing"]) or "-"
            present = ", ".join(category_result["present"]) or "-"
            weak_only = ", ".join(category_result.get("weak_only", [])) or "-"
            lines.append(f"- {category}: present={present}; weak_only={weak_only}; missing={missing}")
        lines.append(f"- forbidden strong leakage: {', '.join(result['forbidden_false_positives']['present']) or '-'}")
        lines.append(f"- forbidden weak-only leakage: {', '.join(result['forbidden_false_positives'].get('weak_only_present', [])) or '-'}")
        lines.append(
            f"- supporting spans: {result['supporting_evidence']['actual_span_count']} / required {result['supporting_evidence']['required_span_count']}"
        )
        lines.append(
            f"- path count: {result['path_presence']['path_count']} total; "
            f"{result['path_presence'].get('meaningful_path_count', 0)} meaningful; "
            f"{result['path_presence'].get('weak_path_count', 0)} weak-only"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def evaluate(graph_dir: Path, dataset_path: Path) -> Dict[str, Any]:
    graph = load_graph(graph_dir)
    cases = load_cases(dataset_path)
    case_results = [evaluate_case(case, graph) for case in cases]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graph_dir": str(graph_dir),
        "dataset": str(dataset_path),
        "graph_counts": {name: len(rows) for name, rows in graph.items()},
        "summary": summarize(case_results),
        "cases": case_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate nerve-centered concept graph completeness.")
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_RESULTS_JSON)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_doc = evaluate(args.graph_dir, args.dataset)
    write_json(args.output_json, results_doc)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(results_doc), encoding="utf-8")
    summary = results_doc["summary"]
    print(
        f"Graph nerve completeness evaluated: {summary['passed_cases']} passed, "
        f"{summary['failed_cases']} failed across {summary['total_cases']} cases."
    )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
