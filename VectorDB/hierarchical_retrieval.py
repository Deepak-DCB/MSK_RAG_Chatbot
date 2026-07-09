#!/usr/bin/env python3
"""Read-only helpers for hierarchical corpus artifacts.

This module does not own retrieval. It augments the existing chunk-first Chroma
pipeline by mapping selected chunk metadata onto reconstructed article, section,
and evidence-span records produced by ``scripts/build_hierarchical_corpus.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = PROJECT_ROOT / "MSKArticlesINDEX" / "hierarchical"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _count_tokens(text: str) -> int:
    return int(round(len(str(text or "").split()) * 1.33))


def _truncate_tokens(text: str, max_tokens: Optional[int]) -> str:
    if not max_tokens or max_tokens <= 0:
        return text or ""
    words = str(text or "").split()
    max_words = max(1, int(max_tokens / 1.33))
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words]).rstrip() + " ..."


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


@dataclass
class HierarchicalCorpus:
    base_dir: Path
    manifest: Dict[str, Any]
    articles: Dict[str, Dict[str, Any]]
    sections: Dict[str, Dict[str, Any]]
    spans: Dict[str, Dict[str, Any]]
    sections_by_article: Dict[str, List[Dict[str, Any]]]
    spans_by_section: Dict[str, List[Dict[str, Any]]]
    article_by_source: Dict[str, Dict[str, Any]]
    section_by_source_name: Dict[tuple, Dict[str, Any]]

    @property
    def available(self) -> bool:
        return bool(self.articles and self.sections and self.spans)

    def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        return self.articles.get(article_id)

    def get_sections_for_article(self, article_id: str) -> List[Dict[str, Any]]:
        return list(self.sections_by_article.get(article_id, []))

    def get_spans_for_section(self, section_id: str) -> List[Dict[str, Any]]:
        return list(self.spans_by_section.get(section_id, []))

    def reconstruct_article_context(self, article_id: str, max_tokens: Optional[int] = None) -> str:
        article = self.get_article(article_id) or {}
        return _truncate_tokens(article.get("reconstructed_text", ""), max_tokens)

    def reconstruct_section_context(self, section_id: str, max_tokens: Optional[int] = None) -> str:
        section = self.sections.get(section_id) or {}
        return _truncate_tokens(section.get("text", ""), max_tokens)

    def map_chunk_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        source = str(meta.get("source_relpath") or "")
        section_name = str(meta.get("section") or "Main")
        article = self.article_by_source.get(source)
        section = self.section_by_source_name.get((source, _norm(section_name)))
        if article and section is None:
            article_sections = self.sections_by_article.get(article["article_id"], [])
            section = article_sections[0] if article_sections else None
        spans = self.get_spans_for_section(section["section_id"]) if section else []
        return {
            "article_id": article.get("article_id") if article else None,
            "section_id": section.get("section_id") if section else None,
            "article": article,
            "section": section,
            "evidence_spans": spans,
        }


_CACHE: Dict[Path, HierarchicalCorpus] = {}


def load_hierarchical_corpus(base_dir: str | Path = DEFAULT_BASE_DIR) -> HierarchicalCorpus:
    base_path = Path(base_dir)
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path
    base_path = base_path.resolve()
    if base_path in _CACHE:
        return _CACHE[base_path]

    required = [
        base_path / "articles.jsonl",
        base_path / "sections.jsonl",
        base_path / "evidence_spans.jsonl",
        base_path / "corpus_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing hierarchical corpus artifacts: " + ", ".join(missing))

    manifest = json.loads((base_path / "corpus_manifest.json").read_text(encoding="utf-8"))
    article_rows = _read_jsonl(base_path / "articles.jsonl")
    section_rows = _read_jsonl(base_path / "sections.jsonl")
    span_rows = _read_jsonl(base_path / "evidence_spans.jsonl")

    articles = {row["article_id"]: row for row in article_rows}
    sections = {row["section_id"]: row for row in section_rows}
    spans = {row["span_id"]: row for row in span_rows}

    sections_by_article: Dict[str, List[Dict[str, Any]]] = {}
    for section in section_rows:
        sections_by_article.setdefault(section["article_id"], []).append(section)
    for items in sections_by_article.values():
        items.sort(key=lambda row: int(row.get("section_order", 0)))

    spans_by_section: Dict[str, List[Dict[str, Any]]] = {}
    for span in span_rows:
        spans_by_section.setdefault(span["section_id"], []).append(span)
    for items in spans_by_section.values():
        items.sort(key=lambda row: int(row.get("span_order", 0)))

    article_by_source = {row["source_relpath"]: row for row in article_rows}
    section_by_source_name = {
        (row["source_relpath"], _norm(row.get("section_name", "Main"))): row
        for row in section_rows
    }

    corpus = HierarchicalCorpus(
        base_dir=base_path,
        manifest=manifest,
        articles=articles,
        sections=sections,
        spans=spans,
        sections_by_article=sections_by_article,
        spans_by_section=spans_by_section,
        article_by_source=article_by_source,
        section_by_source_name=section_by_source_name,
    )
    _CACHE[base_path] = corpus
    return corpus


def get_article(article_id: str) -> Optional[Dict[str, Any]]:
    return load_hierarchical_corpus().get_article(article_id)


def get_sections_for_article(article_id: str) -> List[Dict[str, Any]]:
    return load_hierarchical_corpus().get_sections_for_article(article_id)


def get_spans_for_section(section_id: str) -> List[Dict[str, Any]]:
    return load_hierarchical_corpus().get_spans_for_section(section_id)


def reconstruct_article_context(article_id: str, max_tokens: Optional[int] = None) -> str:
    return load_hierarchical_corpus().reconstruct_article_context(article_id, max_tokens)


def reconstruct_section_context(section_id: str, max_tokens: Optional[int] = None) -> str:
    return load_hierarchical_corpus().reconstruct_section_context(section_id, max_tokens)


def map_chunks_to_hierarchy(selected_context: List[Dict[str, Any]], corpus: Optional[HierarchicalCorpus] = None) -> List[Dict[str, Any]]:
    corpus = corpus or load_hierarchical_corpus()
    mapped: List[Dict[str, Any]] = []
    for item in selected_context:
        meta = item.get("meta", {}) or {}
        mapping = corpus.map_chunk_metadata(meta)
        mapped.append({
            "chunk": item,
            "article_id": mapping.get("article_id"),
            "section_id": mapping.get("section_id"),
            "article": mapping.get("article"),
            "section": mapping.get("section"),
            "evidence_spans": mapping.get("evidence_spans") or [],
        })
    return mapped


def build_citation_map(selected_context: Dict[str, Any] | List[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(selected_context, dict):
        spans = selected_context.get("selected_evidence_spans", []) or []
        sections = selected_context.get("selected_sections", []) or []
        articles = selected_context.get("selected_articles", []) or []
    else:
        spans = []
        sections = []
        articles = []
        for item in selected_context:
            meta = item.get("meta", {}) or {}
            sections.append({
                "source_relpath": meta.get("source_relpath", ""),
                "section_name": meta.get("section", ""),
            })

    by_source_section: Dict[str, List[str]] = {}
    for span in spans:
        key = f"{span.get('source_relpath', '')} — {span.get('section_name', '')}"
        by_source_section.setdefault(key, []).append(span.get("span_id", ""))
    return {
        "span_ids": [span.get("span_id") for span in spans if span.get("span_id")],
        "section_ids": [section.get("section_id") for section in sections if section.get("section_id")],
        "article_ids": [article.get("article_id") for article in articles if article.get("article_id")],
        "by_source_section": by_source_section,
    }


def token_estimate(text: str) -> int:
    return _count_tokens(text)
