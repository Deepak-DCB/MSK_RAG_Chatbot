from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VECTORDB_DIR = PROJECT_ROOT / "VectorDB"
if str(VECTORDB_DIR) not in sys.path:
    sys.path.insert(0, str(VECTORDB_DIR))

from mechanics_retrieval import build_mechanics_context  # noqa: E402
from qaEngine import (  # noqa: E402
    BM25Index,
    QAConfig,
    _backend,
    _red_flag_response,
    _scope_boundary_response,
    build_context_pack,
    compress_context,
    count_tokens,
    detect_red_flags,
    detect_scope_issue,
    group_by_source,
    pick_multichunk_context,
    agentic_run,
)


DEFAULT_MODES = ["normal_ask_default", "mechanics_study"]
COMPARISON_DIR = PROJECT_ROOT / "Evaluation" / "comparisons"
EVIDENCE_SPANS_PATH = PROJECT_ROOT / "MSKArticlesINDEX" / "hierarchical" / "evidence_spans.jsonl"
_SPAN_CACHE: dict[str, dict[str, Any]] | None = None


def safe_slug(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug or "comparison")[:max_len].strip("-") or "comparison"


def preview(text: Any, limit: int = 300) -> str:
    clean = " ".join(str(text or "").split())
    return clean[:limit] + ("..." if len(clean) > limit else "")


def load_questions(path: Path, max_cases: int | None = None) -> list[dict[str, str]]:
    questions: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        q = str(row.get("question") or "").strip()
        if q:
            questions.append({"id": str(row.get("id") or safe_slug(q)), "question": q})
        if max_cases and len(questions) >= max_cases:
            break
    return questions


def load_span_index() -> dict[str, dict[str, Any]]:
    global _SPAN_CACHE
    if _SPAN_CACHE is not None:
        return _SPAN_CACHE
    spans: dict[str, dict[str, Any]] = {}
    if EVIDENCE_SPANS_PATH.exists():
        for line in EVIDENCE_SPANS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            span_id = row.get("span_id")
            if span_id:
                spans[str(span_id)] = row
    _SPAN_CACHE = spans
    return spans


def ask_mode_config(mode: str, *, live_openai: bool) -> QAConfig:
    if mode == "ask_chunk_pack":
        return QAConfig(generate_answer=live_openai, use_reranker=False, context_strategy="chunk_pack")
    elif mode == "ask_hybrid_long_context" or mode == "normal_ask_default":
        return QAConfig(generate_answer=live_openai, use_reranker=False, context_strategy="hybrid_long_context")
    elif mode in {"graph_disabled", "normal_ask_default_graph_disabled", "ask_graph_disabled"}:
        return QAConfig(
            generate_answer=live_openai,
            use_reranker=False,
            context_strategy="hybrid_long_context",
            use_graph_context=False,
            graph_context_strategy="off",
        )
    elif mode in {"graph_enabled", "normal_ask_default_graph_enabled", "ask_graph_enabled"}:
        return QAConfig(
            generate_answer=live_openai,
            use_reranker=False,
            context_strategy="hybrid_long_context",
            use_graph_context=True,
            graph_context_strategy="mechanism_paths",
        )
    return QAConfig(generate_answer=live_openai, use_reranker=False)


def mode_context_strategy(mode: str) -> str:
    if mode == "mechanics_study":
        return "deterministic_mechanics_maps"
    return ask_mode_config(mode, live_openai=False).context_strategy


def compact_chunk(item: dict[str, Any]) -> dict[str, Any]:
    meta = item.get("meta") or {}
    return {
        "chunk_id": meta.get("chunk_id") or meta.get("id") or meta.get("chunk_index"),
        "article": meta.get("title") or meta.get("source") or meta.get("source_relpath"),
        "source": meta.get("source_relpath") or meta.get("source"),
        "section": meta.get("section") or meta.get("section_name"),
        "retrieval_score": item.get("score") or item.get("bm25_score"),
        "distance": item.get("dist"),
        "text_preview": preview(item.get("text")),
    }


def compact_span(span: dict[str, Any]) -> dict[str, Any]:
    return {
        "span_id": span.get("span_id"),
        "article": span.get("title") or span.get("source_relpath"),
        "source": span.get("source_relpath"),
        "section": span.get("section_name"),
        "support_level": span.get("support_level"),
        "text_preview": preview(span.get("text")),
    }


def record_id(record: dict[str, Any]) -> str:
    for key in ("nerve_id", "site_id", "pair_id", "space_id", "chain_id", "id"):
        if record.get(key):
            return str(record[key])
    return str(record.get("name") or record.get("label") or "unknown")


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record_id(record),
        "name": record.get("name") or record.get("question_it_answers") or record_id(record),
        "support_level": record.get("support_level"),
        "summary": preview(
            record.get("course_summary")
            or record.get("mechanical_trigger")
            or record.get("mechanical_role")
            or record.get("question_it_answers")
            or record.get("notes")
        ),
        "evidence_span_ids": record.get("evidence_span_ids") or [],
    }


def collect_mechanics_span_ids(ctx: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in ("nerves", "entrapment_sites", "muscle_pairs", "spaces", "mechanism_chains"):
        for record in ctx.get(key, []) or []:
            for span_id in record.get("evidence_span_ids", []) or []:
                if span_id and span_id not in seen:
                    seen.add(span_id)
                    out.append(str(span_id))
    return out


def mechanics_span_support(ctx: dict[str, Any]) -> dict[str, str]:
    support: dict[str, str] = {}
    for key in ("nerves", "entrapment_sites", "muscle_pairs", "spaces", "mechanism_chains"):
        for record in ctx.get(key, []) or []:
            level = record.get("support_level")
            for span_id in record.get("evidence_span_ids", []) or []:
                if span_id and level and span_id not in support:
                    support[str(span_id)] = str(level)
    return support


def compact_mechanics_spans(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    index = load_span_index()
    support = mechanics_span_support(ctx)
    spans: list[dict[str, Any]] = []
    for span_id in collect_mechanics_span_ids(ctx):
        row = dict(index.get(span_id) or {})
        row.setdefault("span_id", span_id)
        if support.get(span_id):
            row["support_level"] = support[span_id]
        spans.append(compact_span(row))
    return spans


def deterministic_mechanics_answer(question: str, ctx: dict[str, Any]) -> str:
    if not ctx.get("available"):
        reason = ctx.get("fallback_reason") or "mechanics_artifacts_unavailable"
        return (
            "**Short answer**\n"
            f"The mechanics study map is not available right now (`{reason}`).\n\n"
            "**Safety / interpretation boundary**\n"
            "This study mode is for article interpretation only. It cannot diagnose symptoms, rule out urgent problems, or prescribe treatment."
        )
    records = []
    for key in ("nerves", "entrapment_sites", "muscle_pairs", "spaces", "mechanism_chains"):
        records.extend(ctx.get(key, []) or [])
    direct = [r for r in records if r.get("support_level") == "direct"]
    uncertain = [r for r in records if r.get("support_level") != "direct"]
    chains = ctx.get("mechanism_chains", []) or []
    chain_lines = []
    for chain in chains[:3]:
        steps = chain.get("steps") or []
        if steps:
            chain_lines.append(f"- {record_id(chain)} ({chain.get('support_level', 'unknown')}): " + " -> ".join(steps))
    direct_lines = [f"- {record_id(r)}: {preview(r.get('mechanical_role') or r.get('course_summary') or r.get('question_it_answers'))}" for r in direct[:8]]
    uncertain_lines = [f"- {record_id(r)} ({r.get('support_level', 'unknown')}): {preview(r.get('notes') or r.get('weakest_step') or r.get('unsupported_or_uncertain_notes') or r.get('mechanical_role') or r.get('question_it_answers'))}" for r in uncertain[:8]]
    if direct:
        matched_summary = ", ".join(record_id(r) for r in direct[:3]) or "matched structures"
        short = f"The current mechanics map has direct support for {matched_summary} and indirect support for mechanism chains."
    elif uncertain:
        short = "The current mechanics map only gives indirect or uncertain support for this question; read as a study aid, not a conclusion."
    else:
        short = "The current mechanics map does not contain enough matching records to answer this beyond noting the evidence gap."
    return "\n\n".join(
        [
            "**Short answer**\n" + short,
            "**Mechanism chain**\n" + ("\n".join(chain_lines) if chain_lines else "No matching mechanism chain was found in the current mechanics map."),
            "**Directly supported claims**\n" + ("\n".join(direct_lines) if direct_lines else "No directly supported matching claim was found."),
            "**Indirect or uncertain links**\n" + ("\n".join(uncertain_lines) if uncertain_lines else "No indirect/uncertain matching link was found."),
            "**What the corpus does not prove**\n- It does not prove that this mechanism explains any specific person's symptoms.\n- It does not establish a diagnosis or treatment plan.",
            "**Safety / interpretation boundary**\nThis mode is for learning and article interpretation only. New or worsening neurologic symptoms or other red flags should be assessed in person urgently.",
            "**Evidence spans used**\n" + ("\n".join(f"- {sid}" for sid in collect_mechanics_span_ids(ctx)) or "No evidence span IDs were available."),
        ]
    )


def run_ask_dry(question: str, mode: str) -> dict[str, Any]:
    started = time.time()
    red_flags = detect_red_flags(question)
    if red_flags:
        return base_mode_result(question, mode, answer=_red_flag_response(red_flags), safety=True, safety_reasons=red_flags, fallback="safety_gate_triggered", answer_type="safety_refusal")
    scope_issue = detect_scope_issue(question)
    if scope_issue:
        return base_mode_result(question, mode, answer=_scope_boundary_response(scope_issue), scope_issue=scope_issue, fallback="scope_boundary", answer_type="scope_boundary")

    cfg = ask_mode_config(mode, live_openai=False)
    try:
        collection = _backend.load_collection()
        bm25 = BM25Index.get()
        bm25._build(collection)
        bm25_results = bm25.search(question, top_n=max(10, cfg.retrieval_pool))
        candidates = [
            {"text": item["text"], "meta": item.get("meta") or {}, "dist": 1.0 / max(float(item.get("bm25_score") or 1.0), 0.0001), "score": item.get("bm25_score")}
            for item in bm25_results
        ]
        grouped = group_by_source(candidates)
        pooled: list[dict[str, Any]] = []
        for group in grouped.values():
            pooled.extend(sorted(group, key=lambda row: row.get("dist", 1.0))[: cfg.per_source_pool])
        context = pick_multichunk_context(
            pooled[: cfg.final_limit],
            top_k=cfg.top_k,
            per_source_max=cfg.per_source_max,
            budget_tokens=cfg.budget_tokens,
            neighbor_headroom=cfg.neighbor_headroom,
        )
        context = compress_context(context, question)
        pack = build_context_pack(context, question, cfg)
        return normalize_ask_result(
            question,
            mode,
            {
                "answer": "",
                "contexts": context,
                "original_question": question,
                "refined_query": question,
                "context_strategy": pack.strategy,
                "fallback_reason": pack.fallback_reason,
                "selected_articles": pack.selected_articles,
                "selected_sections": pack.selected_sections,
                "selected_evidence_spans": pack.selected_evidence_spans,
                "context_token_estimate": pack.token_estimate,
                "total_context_token_estimate": pack.total_context_token_estimate or pack.token_estimate,
                "graph_available": pack.graph_available,
                "graph_fallback_reason": pack.graph_fallback_reason,
                "graph_paths": pack.graph_paths,
                "graph_edges": pack.graph_edges,
                "graph_supporting_spans": pack.graph_supporting_spans,
                "retrieval_time": time.time() - started,
                "generation_time": 0.0,
                "model_name": cfg.openai_model,
            },
            answer_type="dry_run_retrieval_only",
            live=False,
        )
    except Exception as exc:
        return base_mode_result(question, mode, fallback=f"dry_run_context_error: {exc}", answer_type="dry_run_error")


def run_ask_live(question: str, mode: str) -> dict[str, Any]:
    started = time.time()
    cfg = ask_mode_config(mode, live_openai=True)
    res = agentic_run(question, cfg=cfg)
    res["latency_seconds"] = time.time() - started
    res["model_name"] = cfg.openai_model
    return normalize_ask_result(question, mode, res, answer_type="openai_generated", live=True)


def run_mechanics(question: str, mode: str) -> dict[str, Any]:
    started = time.time()
    red_flags = detect_red_flags(question)
    if red_flags:
        return base_mode_result(question, mode, answer=_red_flag_response(red_flags), safety=True, safety_reasons=red_flags, fallback="safety_gate_triggered", answer_type="safety_refusal")
    scope_issue = detect_scope_issue(question)
    if scope_issue:
        return base_mode_result(question, mode, answer=_scope_boundary_response(scope_issue), scope_issue=scope_issue, fallback="scope_boundary", answer_type="scope_boundary")
    ctx = build_mechanics_context(question, max_items=8)
    records = {
        "nerves": [compact_record(r) for r in ctx.get("nerves", []) or []],
        "entrapment_sites": [compact_record(r) for r in ctx.get("entrapment_sites", []) or []],
        "muscle_pairs": [compact_record(r) for r in ctx.get("muscle_pairs", []) or []],
        "spaces": [compact_record(r) for r in ctx.get("spaces", []) or []],
        "mechanism_chains": [compact_record(r) for r in ctx.get("mechanism_chains", []) or []],
    }
    return {
        "mode": mode,
        "answer_type": "deterministic_mechanics",
        "original_question": question,
        "refined_query": question,
        "context_strategy": "deterministic_mechanics_maps",
        "graph_enabled": False,
        "mechanics_enabled": True,
        "selected_articles": [],
        "selected_sections": [],
        "selected_chunks": [],
        "selected_evidence_spans": compact_mechanics_spans(ctx),
        "selected_graph_paths": [],
        "selected_graph_edges": [],
        "selected_mechanics_records": records,
        "context_token_estimate": count_tokens(ctx.get("context", "")),
        "fallback_reason": ctx.get("fallback_reason") or None,
        "safety_gate_triggered": False,
        "safety_gate_reasons": [],
        "scope_issue": None,
        "answer": deterministic_mechanics_answer(question, ctx),
        "latency_seconds": time.time() - started,
        "model_name": None,
    }


def base_mode_result(
    question: str,
    mode: str,
    *,
    answer: str = "",
    safety: bool = False,
    safety_reasons: list[str] | None = None,
    scope_issue: str | None = None,
    fallback: str | None = None,
    answer_type: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "answer_type": answer_type,
        "original_question": question,
        "refined_query": question,
        "context_strategy": mode_context_strategy(mode),
        "graph_enabled": mode != "mechanics_study",
        "mechanics_enabled": mode == "mechanics_study",
        "selected_articles": [],
        "selected_sections": [],
        "selected_chunks": [],
        "selected_evidence_spans": [],
        "selected_graph_paths": [],
        "selected_graph_edges": [],
        "selected_mechanics_records": {"nerves": [], "entrapment_sites": [], "muscle_pairs": [], "spaces": [], "mechanism_chains": []},
        "context_token_estimate": 0,
        "fallback_reason": fallback,
        "safety_gate_triggered": safety,
        "safety_gate_reasons": safety_reasons or [],
        "scope_issue": scope_issue,
        "answer": answer,
        "latency_seconds": 0.0,
        "model_name": None,
    }


def normalize_ask_result(question: str, mode: str, res: dict[str, Any], *, answer_type: str, live: bool) -> dict[str, Any]:
    cfg = ask_mode_config(mode, live_openai=live)
    return {
        "mode": mode,
        "answer_type": answer_type if res.get("answer") or not res.get("safety_gate_triggered") else "safety_refusal",
        "original_question": res.get("original_question") or question,
        "refined_query": res.get("refined_query") or question,
        "context_strategy": res.get("context_strategy") or cfg.context_strategy,
        "graph_enabled": bool(cfg.use_graph_context),
        "mechanics_enabled": False,
        "selected_articles": res.get("selected_articles", []) or [],
        "selected_sections": res.get("selected_sections", []) or [],
        "selected_chunks": [compact_chunk(item) for item in (res.get("contexts", []) or [])],
        "selected_evidence_spans": [compact_span(span) for span in (res.get("selected_evidence_spans") or res.get("evidence_spans") or [])],
        "selected_graph_paths": res.get("graph_paths", []) or [],
        "selected_graph_edges": res.get("graph_edges", []) or [],
        "selected_mechanics_records": {"nerves": [], "entrapment_sites": [], "muscle_pairs": [], "spaces": [], "mechanism_chains": []},
        "context_token_estimate": res.get("total_context_token_estimate") or res.get("context_token_estimate") or res.get("context_tokens") or 0,
        "fallback_reason": res.get("fallback_reason") or res.get("graph_fallback_reason"),
        "safety_gate_triggered": bool(res.get("safety_gate_triggered", False)),
        "safety_gate_reasons": res.get("safety_gate_reasons", []) or [],
        "scope_issue": res.get("scope_issue"),
        "answer": res.get("answer", "") or "",
        "latency_seconds": res.get("latency_seconds") or (float(res.get("retrieval_time") or 0) + float(res.get("generation_time") or 0)),
        "model_name": res.get("model_name") or cfg.openai_model,
        "retrieval_confidence": res.get("retrieval_confidence"),
    }


def run_mode(question: str, mode: str, *, live_openai: bool) -> dict[str, Any]:
    if mode == "mechanics_study":
        return run_mechanics(question, mode)
    if live_openai:
        return run_ask_live(question, mode)
    return run_ask_dry(question, mode)


def mode_strengths(result: dict[str, Any]) -> list[str]:
    strengths = []
    if result.get("selected_evidence_spans"):
        strengths.append("Exposes selected evidence spans for grounding review.")
    if result.get("selected_graph_paths"):
        strengths.append("Includes graph paths for mechanism traceability.")
    records = result.get("selected_mechanics_records") or {}
    if sum(len(v or []) for v in records.values()) > 0:
        strengths.append("Separates deterministic mechanics records and support levels.")
    if result.get("answer"):
        strengths.append("Provides final answer text for side-by-side review.")
    return strengths or ["No clear strength surfaced in this run."]


def mode_weaknesses(result: dict[str, Any]) -> list[str]:
    weaknesses = []
    if result.get("answer_type") == "dry_run_retrieval_only":
        weaknesses.append("Dry-run mode does not generate a final OpenAI answer.")
    if result.get("fallback_reason"):
        weaknesses.append(f"Fallback or limitation: {result.get('fallback_reason')}.")
    if not result.get("selected_evidence_spans") and not result.get("selected_chunks"):
        weaknesses.append("No selected corpus chunks or evidence spans were available for this mode.")
    if result.get("mechanics_enabled") and not result.get("selected_mechanics_records"):
        weaknesses.append("No mechanics records were selected.")
    return weaknesses or ["No obvious weakness surfaced; review answer content manually."]


def count_mechanics_records(result: dict[str, Any]) -> int:
    records = result.get("selected_mechanics_records") or {}
    return sum(len(records.get(key) or []) for key in ("nerves", "entrapment_sites", "muscle_pairs", "spaces", "mechanism_chains"))


def md_list(items: list[Any], empty: str = "None selected.") -> str:
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


def render_selected_info(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("**Articles**")
    articles = [f"{a.get('article_id') or 'unknown'}: {a.get('title') or a.get('source_relpath') or 'untitled'}" for a in result.get("selected_articles", [])]
    lines.append(md_list(articles))
    lines.append("\n**Sections**")
    sections = [f"{s.get('section_id') or 'unknown'}: {s.get('section_name') or s.get('title') or 'untitled'}" for s in result.get("selected_sections", [])]
    lines.append(md_list(sections))
    lines.append("\n**Chunks**")
    chunks = []
    for chunk in result.get("selected_chunks", [])[:8]:
        chunks.append(f"{chunk.get('chunk_id') or 'unknown'} | {chunk.get('article') or chunk.get('source') or 'unknown'} | {chunk.get('section') or 'n/a'} | score={chunk.get('retrieval_score') or chunk.get('distance') or 'n/a'} | {chunk.get('text_preview')}")
    lines.append(md_list(chunks))
    lines.append("\n**Evidence Spans**")
    spans = []
    for span in result.get("selected_evidence_spans", [])[:12]:
        spans.append(f"{span.get('span_id') or 'unknown'} | {span.get('article') or span.get('source') or 'unknown'} | {span.get('section') or 'n/a'} | support={span.get('support_level') or 'n/a'} | {span.get('text_preview') or ''}")
    lines.append(md_list(spans))
    lines.append("\n**Graph Paths / Edges**")
    graph_items = [preview(path.get("path_label") or path.get("name") or path.get("path_id") or path) for path in result.get("selected_graph_paths", [])[:8]]
    graph_items.extend(preview(edge.get("edge_id") or edge.get("claim") or edge) for edge in result.get("selected_graph_edges", [])[:5])
    lines.append(md_list(graph_items))
    lines.append("\n**Mechanics Records**")
    mechanics_items = []
    records = result.get("selected_mechanics_records") or {}
    for key in ("nerves", "entrapment_sites", "muscle_pairs", "spaces", "mechanism_chains"):
        for record in records.get(key, []) or []:
            mechanics_items.append(f"{key}: {record.get('id')} | support={record.get('support_level') or 'n/a'} | {record.get('summary') or ''}")
    lines.append(md_list(mechanics_items))
    return "\n".join(lines)


def render_markdown(question: str, results: list[dict[str, Any]], *, live_openai: bool) -> str:
    table = [
        "| Mode | Answer type | Articles selected | Evidence spans selected | Graph paths selected | Mechanics records selected | Token estimate | Fallback | Safety triggered |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for result in results:
        table.append(
            "| {mode} | {answer_type} | {articles} | {spans} | {paths} | {mechanics} | {tokens} | {fallback} | {safety} |".format(
                mode=result.get("mode"),
                answer_type=result.get("answer_type"),
                articles=len(result.get("selected_articles") or []),
                spans=len(result.get("selected_evidence_spans") or []),
                paths=len(result.get("selected_graph_paths") or []),
                mechanics=count_mechanics_records(result),
                tokens=result.get("context_token_estimate") or 0,
                fallback=result.get("fallback_reason") or "",
                safety="yes" if result.get("safety_gate_triggered") else "no",
            )
        )
    parts = [
        "# Answer Mode Comparison",
        "",
        "Question:",
        question,
        "",
        "## Summary Table",
        "",
        "\n".join(table),
    ]
    for idx, result in enumerate(results, start=1):
        parts.extend(
            [
                "",
                f"## Mode {idx}: {result.get('mode')}",
                "",
                "### Retrieval process",
                "",
                f"- Original question: {result.get('original_question')}",
                f"- Rewritten/refined query: {result.get('refined_query')}",
                f"- Context strategy: {result.get('context_strategy')}",
                f"- Graph enabled: {result.get('graph_enabled')}",
                f"- Mechanics enabled: {result.get('mechanics_enabled')}",
                f"- Model: {result.get('model_name') or 'n/a'}",
                f"- Latency seconds: {round(float(result.get('latency_seconds') or 0), 3)}",
                f"- Fallback reason: {result.get('fallback_reason') or 'none'}",
                f"- Safety triggered: {result.get('safety_gate_triggered')}",
                "",
                "### Selected information",
                "",
                render_selected_info(result),
                "",
                "### Final answer",
                "",
                result.get("answer") or ("Dry-run only: no final OpenAI answer was requested." if not live_openai else "No answer text returned."),
                "",
                "### Strengths",
                "",
                md_list(mode_strengths(result)),
                "",
                "### Weaknesses",
                "",
                md_list(mode_weaknesses(result)),
            ]
        )
    parts.extend(["", "## Cross-mode comparison", ""])
    normal = next((r for r in results if r.get("mode") != "mechanics_study"), None)
    mechanics = next((r for r in results if r.get("mode") == "mechanics_study"), None)
    if normal and mechanics:
        normal_articles = {a.get("title") or a.get("source_relpath") for a in normal.get("selected_articles", [])}
        mechanics_record_ids = []
        for records in (mechanics.get("selected_mechanics_records") or {}).values():
            mechanics_record_ids.extend(r.get("id") for r in records or [])
        parts.extend(
            [
                f"- What normal ask found that mechanics study missed: article/section context from {', '.join(sorted(x for x in normal_articles if x)[:5]) or 'none selected'}.",
                f"- What mechanics study found that normal ask missed: mechanics records {', '.join(mechanics_record_ids[:8]) or 'none selected'}.",
                "- Which answer is more useful for body-mechanics reasoning: review mechanics_study when deterministic structure/support separation matters; review normal ask when full corpus prose synthesis matters.",
                "- Which answer is more evidence-grounded: compare the evidence span counts and inspect selected span previews; higher count alone is not proof of better grounding.",
                "- Which answer is more cautious about unsupported claims: prefer the mode that explicitly labels indirect/uncertain links and avoids unsupported diagnosis or treatment claims.",
            ]
        )
    else:
        parts.extend(
            [
                "- What normal ask found that mechanics study missed: not evaluated because both mode families were not run.",
                "- What mechanics study found that normal ask missed: not evaluated because both mode families were not run.",
                "- Which answer is more useful for body-mechanics reasoning: requires manual review of the selected modes.",
                "- Which answer is more evidence-grounded: requires manual review of selected chunks and spans.",
                "- Which answer is more cautious about unsupported claims: requires manual review of support labels and final answer wording.",
            ]
        )
    return "\n".join(parts) + "\n"


def write_reports(question: str, results: list[dict[str, Any]], out_path: Path, json_path: Path, *, live_openai: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "question": question,
        "live_openai": live_openai,
        "created_at_unix": time.time(),
        "results": results,
    }
    out_path.write_text(render_markdown(question, results, live_openai=live_openai), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_modes(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_MODES
    return [mode.strip() for mode in value.split(",") if mode.strip()]


def default_paths(question: str) -> tuple[Path, Path]:
    slug = safe_slug(question)
    return COMPARISON_DIR / f"{slug}.md", COMPARISON_DIR / f"{slug}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare MSK answer modes for one question at a time.")
    parser.add_argument("--question", help="Question to compare.")
    parser.add_argument("--questions-file", type=Path, help="Optional JSONL question file for bounded batch comparison.")
    parser.add_argument("--max-cases", type=int, default=1, help="Maximum questions to run from --questions-file.")
    parser.add_argument("--out", type=Path, help="Markdown report path.")
    parser.add_argument("--json-out", type=Path, help="JSON report path.")
    parser.add_argument("--live-openai", action="store_true", help="Call the real OpenAI-backed ask path for ask modes.")
    parser.add_argument("--dry-run", action="store_true", help="Collect retrieval/context metadata without OpenAI generation.")
    parser.add_argument("--modes", help="Comma-separated modes to run.")
    args = parser.parse_args(argv)

    if args.live_openai and args.dry_run:
        parser.error("Use either --live-openai or --dry-run, not both.")

    live_openai = bool(args.live_openai)
    modes = parse_modes(args.modes)
    ask_modes = [mode for mode in modes if mode != "mechanics_study"]
    if live_openai and ask_modes and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot run --live-openai ask modes. Use --dry-run or set the key.", file=sys.stderr)
        return 2

    questions: list[dict[str, str]] = []
    if args.questions_file:
        questions = load_questions(args.questions_file, max_cases=max(1, args.max_cases or 1))
    elif args.question:
        questions = [{"id": safe_slug(args.question), "question": args.question}]
    else:
        parser.error("Provide --question or --questions-file.")

    for idx, item in enumerate(questions):
        question = item["question"]
        default_md, default_json = default_paths(question)
        if len(questions) == 1:
            out_path = args.out or default_md
            json_path = args.json_out or default_json
        else:
            stem = safe_slug(item.get("id") or question)
            out_path = args.out.with_name(f"{args.out.stem}-{stem}.md") if args.out else COMPARISON_DIR / f"{stem}.md"
            json_path = args.json_out.with_name(f"{args.json_out.stem}-{stem}.json") if args.json_out else COMPARISON_DIR / f"{stem}.json"
        results = [run_mode(question, mode, live_openai=live_openai) for mode in modes]
        write_reports(question, results, out_path, json_path, live_openai=live_openai)
        print(f"Wrote markdown report: {out_path}")
        print(f"Wrote JSON report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
