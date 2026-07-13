from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from graph_vocab import SCHEMA_VERSION, alias_patterns, all_entities, detect_entities


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "MSKArticlesINDEX" / "graph"
DEFAULT_HIERARCHICAL_DIR = PROJECT_ROOT / "MSKArticlesINDEX" / "hierarchical"
SUPPORT_SCORE = {"direct": 5, "indirect": 4, "inferred_from_same_section": 3, "inferred_from_path": 2, "weak": 1, "unsupported": 0}
CLAIM_SCORE = {"strong": 4, "moderate": 3, "weak": 2, "speculative": 1}

# Parsed artifacts are static at runtime but cost ~8MB of JSON parsing per load, which
# previously happened on every request. Cache by (path, mtime, size) so a rebuilt
# artifact is picked up without a restart.
_GRAPH_CACHE: Dict[Any, Dict[str, Any]] = {}
_SPAN_CACHE: Dict[Any, Dict[str, Dict[str, Any]]] = {}


def _cache_key(path: Path) -> Any:
    try:
        st = path.stat()
        return (str(path.resolve()), st.st_mtime_ns, st.st_size)
    except OSError:
        return (str(path), None, None)


def _is_loose_co_mention(edge: Dict[str, Any]) -> bool:
    return str(edge.get("relation_type")) == "mentioned_with" and str(edge.get("support_level")) == "weak"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Parse a JSONL artifact, skipping (and logging) individual malformed rows.

    A single corrupt row must not take down the whole graph: these are generated files,
    and one bad line previously raised JSONDecodeError out of load_graph, silently
    disabling graph context everywhere.

    Two distinct failure modes are skipped here:
      * unparseable JSON (JSONDecodeError) — e.g. a stray keystroke saved into the file;
      * *valid* JSON that is not an object — a bare string/number/list/null. These parse
        fine, so they used to survive into `nodes`, and then blew up as an AttributeError
        on `node.get(...)` in load_graph — outside its try/except — taking the graph down
        by a different route.
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skipping unparseable line %d in %s: %s", lineno, path.name, exc)
                continue
            if not isinstance(row, dict):
                logger.warning(
                    "skipping non-object line %d in %s: expected a JSON object, got %s",
                    lineno, path.name, type(row).__name__,
                )
                continue
            rows.append(row)
    return rows


def _token_estimate(text: str) -> int:
    return max(0, int(len((text or "").split()) * 1.33))


def load_graph(base_dir: str | Path = DEFAULT_GRAPH_DIR) -> Dict[str, Any]:
    base = Path(base_dir)
    required = ["nodes.jsonl", "edges.jsonl", "paths.jsonl", "claims.jsonl", "graph_manifest.json"]
    missing = [name for name in required if not (base / name).exists()]
    if missing:
        return {"available": False, "fallback_reason": "graph_artifacts_missing", "missing": missing}

    cache_key = tuple(_cache_key(base / name) for name in required)
    cached = _GRAPH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        nodes = _read_jsonl(base / "nodes.jsonl")
        edges = _read_jsonl(base / "edges.jsonl")
        paths = _read_jsonl(base / "paths.jsonl")
        claims = _read_jsonl(base / "claims.jsonl")
        manifest = json.loads((base / "graph_manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "fallback_reason": f"graph_load_error:{type(exc).__name__}"}

    if not nodes:
        return {"available": False, "fallback_reason": "graph_nodes_empty"}

    node_by_id = {str(node.get("node_id")): node for node in nodes}
    edges_by_node: Dict[str, List[Dict[str, Any]]] = {}
    edge_by_id = {str(edge.get("edge_id")): edge for edge in edges}
    paths_by_node: Dict[str, List[Dict[str, Any]]] = {}
    for edge in edges:
        edges_by_node.setdefault(str(edge.get("source_node_id")), []).append(edge)
        edges_by_node.setdefault(str(edge.get("target_node_id")), []).append(edge)
    for path in paths:
        for node_id in path.get("node_ids") or []:
            paths_by_node.setdefault(str(node_id), []).append(path)

    graph = {
        "available": True,
        "fallback_reason": None,
        "base_dir": str(base),
        "nodes": nodes,
        "edges": edges,
        "paths": paths,
        "claims": claims,
        "manifest": manifest,
        "node_by_id": node_by_id,
        "edge_by_id": edge_by_id,
        "edges_by_node": edges_by_node,
        "paths_by_node": paths_by_node,
    }
    _GRAPH_CACHE[cache_key] = graph
    return graph


def find_nodes(query: str, graph: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    graph = graph or load_graph()
    if not graph.get("available"):
        return []
    query_l = (query or "").lower()
    detected = detect_entities(query_l)
    out = []
    for node in graph.get("nodes", []):
        name = str(node.get("canonical_name") or "")
        aliases = node.get("aliases") or []
        if name in detected or any(pattern.search(query_l) for pattern in alias_patterns(aliases)):
            score = 2 if name in detected else 1
            node = dict(node)
            node["query_overlap_score"] = score
            out.append(node)
    return sorted(out, key=lambda n: (-int(n.get("query_overlap_score", 0)), str(n.get("canonical_name"))))


def find_edges_for_node(node_id: str, graph: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    graph = graph or load_graph()
    if not graph.get("available"):
        return []
    return list(graph.get("edges_by_node", {}).get(node_id, []))


def find_paths_for_nodes(node_ids: Iterable[str], graph: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    graph = graph or load_graph()
    if not graph.get("available"):
        return []
    wanted = set(str(node_id) for node_id in node_ids)
    paths = []
    for path in graph.get("paths", []):
        overlap = len(wanted.intersection(set(path.get("node_ids") or [])))
        if overlap:
            item = dict(path)
            item["query_node_overlap"] = overlap
            paths.append(item)
    return sorted(
        paths,
        key=lambda p: (
            -int(p.get("query_node_overlap", 0)),
            -SUPPORT_SCORE.get(str(p.get("weakest_support_level")), 0),
            len(p.get("node_ids") or []),
        ),
    )


def _load_spans_by_id(hierarchical_dir: str | Path = DEFAULT_HIERARCHICAL_DIR) -> Dict[str, Dict[str, Any]]:
    path = Path(hierarchical_dir) / "evidence_spans.jsonl"
    if not path.exists():
        return {}
    cache_key = _cache_key(path)
    cached = _SPAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    spans = {str(row.get("span_id")): row for row in _read_jsonl(path)}
    _SPAN_CACHE[cache_key] = spans
    return spans


def get_supporting_spans_for_path(path_id: str, graph: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    graph = graph or load_graph()
    if not graph.get("available"):
        return []
    path = next((p for p in graph.get("paths", []) if p.get("path_id") == path_id), None)
    if not path:
        return []
    span_by_id = _load_spans_by_id()
    return [span_by_id[sid] for sid in (path.get("source_span_ids") or []) if sid in span_by_id]


def _public_node(node: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "canonical_name": node.get("canonical_name"),
        "node_type": node.get("node_type"),
        "confidence": node.get("confidence"),
    }


def _public_edge(edge: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    node_by_id = graph.get("node_by_id", {})
    return {
        "edge_id": edge.get("edge_id"),
        "source_node_id": edge.get("source_node_id"),
        "target_node_id": edge.get("target_node_id"),
        "source": (node_by_id.get(edge.get("source_node_id")) or {}).get("canonical_name"),
        "target": (node_by_id.get(edge.get("target_node_id")) or {}).get("canonical_name"),
        "relation_type": edge.get("relation_type"),
        "support_level": edge.get("support_level"),
        "claim_strength": edge.get("claim_strength"),
        "clinical_risk": edge.get("clinical_risk"),
        "source_span_ids": edge.get("source_span_ids") or [],
    }


def _public_path(path: Dict[str, Any], graph: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "path_id": path.get("path_id"),
        "path_family": path.get("path_family"),
        "path_text": path.get("path_text"),
        "node_ids": path.get("node_ids") or [],
        "edge_ids": path.get("edge_ids") or [],
        "path_support": path.get("path_support"),
        "weakest_support_level": path.get("weakest_support_level"),
        "clinical_policy": path.get("clinical_policy"),
        "path_source_scope": path.get("path_source_scope"),
        "source_span_ids": path.get("source_span_ids") or [],
    }


def _edge_score(edge: Dict[str, Any], query_node_ids: Set[str]) -> tuple:
    overlap = int(edge.get("source_node_id") in query_node_ids) + int(edge.get("target_node_id") in query_node_ids)
    return (
        -overlap,
        -SUPPORT_SCORE.get(str(edge.get("support_level")), 0),
        -CLAIM_SCORE.get(str(edge.get("claim_strength")), 0),
        str(edge.get("relation_type")),
    )


def build_graph_context(
    query: str,
    max_paths: int = 5,
    max_edges: int = 20,
    max_spans: int = 8,
    max_graph_tokens: int = 1800,
    base_dir: str | Path = DEFAULT_GRAPH_DIR,
) -> Dict[str, Any]:
    graph = load_graph(base_dir)
    empty = {
        "available": False,
        "fallback_reason": graph.get("fallback_reason") or "graph_unavailable",
        "nodes": [],
        "edges": [],
        "paths": [],
        "supporting_spans": [],
        "context_token_estimate": 0,
        "context": "",
        "schema_version": SCHEMA_VERSION,
    }
    if not graph.get("available"):
        return empty

    nodes = find_nodes(query, graph)
    if not nodes:
        empty.update({"available": True, "fallback_reason": "no_query_node_match"})
        return empty

    node_ids = {str(node.get("node_id")) for node in nodes}
    paths = find_paths_for_nodes(node_ids, graph)[:max_paths]
    path_edge_ids = {edge_id for path in paths for edge_id in (path.get("edge_ids") or [])}
    connected_edges = [edge for node_id in node_ids for edge in find_edges_for_node(node_id, graph)]
    edge_by_id: Dict[str, Dict[str, Any]] = {str(edge.get("edge_id")): edge for edge in connected_edges}
    for edge_id in path_edge_ids:
        if edge_id in graph.get("edge_by_id", {}):
            edge_by_id[edge_id] = graph["edge_by_id"][edge_id]
    edges = sorted(
        [edge for edge_id, edge in edge_by_id.items() if edge_id in path_edge_ids or not _is_loose_co_mention(edge)],
        key=lambda e: _edge_score(e, node_ids),
    )[:max_edges]

    span_ids: List[str] = []
    for path in paths:
        span_ids.extend(path.get("source_span_ids") or [])
    for edge in edges:
        span_ids.extend(edge.get("source_span_ids") or [])
    seen = set()
    span_ids = [sid for sid in span_ids if not (sid in seen or seen.add(sid))]
    span_by_id = _load_spans_by_id()
    supporting_spans = [span_by_id[sid] for sid in span_ids if sid in span_by_id][:max_spans]

    pack = {
        "available": True,
        "fallback_reason": None,
        "nodes": [_public_node(n) for n in nodes[:12]],
        "edges": [_public_edge(e, graph) for e in edges],
        "paths": [_public_path(p, graph) for p in paths],
        "supporting_spans": supporting_spans,
        "schema_version": SCHEMA_VERSION,
    }
    context = format_graph_context(pack)
    if _token_estimate(context) > max_graph_tokens:
        while supporting_spans and _token_estimate(context) > max_graph_tokens:
            supporting_spans.pop()
            pack["supporting_spans"] = supporting_spans
            context = format_graph_context(pack)
        while pack["edges"] and _token_estimate(context) > max_graph_tokens:
            pack["edges"].pop()
            context = format_graph_context(pack)
    pack["context"] = context
    pack["context_token_estimate"] = min(_token_estimate(context), max_graph_tokens)
    return pack


def format_graph_context(graph_pack: Dict[str, Any]) -> str:
    if not graph_pack.get("available"):
        return ""
    lines = ["MECHANISM GRAPH CONTEXT", "Use only with supporting evidence spans; do not collapse indirect paths into direct causal claims."]
    nodes = graph_pack.get("nodes") or []
    if nodes:
        lines.append("Matched concepts: " + ", ".join(str(n.get("canonical_name")) for n in nodes[:10]))
    paths = graph_pack.get("paths") or []
    if paths:
        lines.append("Mechanism paths:")
        for path in paths[:5]:
            lines.append(
                f"- {path.get('path_text')} | support={path.get('path_support')} | "
                f"weakest={path.get('weakest_support_level')} | policy={path.get('clinical_policy')}"
            )
    edges = graph_pack.get("edges") or []
    if edges:
        lines.append("Key supported edges:")
        for edge in edges[:12]:
            lines.append(
                f"- {edge.get('source')} --{edge.get('relation_type')}--> {edge.get('target')} "
                f"| support={edge.get('support_level')} | strength={edge.get('claim_strength')}"
            )
    spans = graph_pack.get("supporting_spans") or []
    if spans:
        lines.append("Graph-supporting evidence spans:")
        for i, span in enumerate(spans[:8], start=1):
            text = re.sub(r"\s+", " ", str(span.get("text") or "")).strip()
            if len(text) > 420:
                text = text[:420].rstrip() + " ..."
            lines.append(f"[{i}] {span.get('title','')} · {span.get('section_name','')} ({span.get('source_relpath','')})\n{text}")
    return "\n".join(lines).strip()
