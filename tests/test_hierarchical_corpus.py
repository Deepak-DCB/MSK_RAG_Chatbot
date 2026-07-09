from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_hierarchical_corpus import build_hierarchical_corpus


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_hierarchical_corpus_outputs_valid_deterministic_artifacts(tmp_path):
    rows = [
        {
            "article_id": "article-a",
            "chunk_id": "chunk-a1",
            "title": "Article A",
            "section": "Main",
            "chunk_idx": "0",
            "article_seq": 0,
            "body": "First sentence explains the neck. Second sentence adds useful context. Third sentence supports citation.",
            "embed_text": "Article A Main First sentence explains the neck.",
            "source_relpath": "source/a/index.html",
            "token_len": 20,
            "word_len": 13,
        },
        {
            "article_id": "article-a",
            "chunk_id": "chunk-a2",
            "title": "Article A",
            "section": "Mechanism",
            "chunk_idx": "1",
            "article_seq": 1,
            "body": "Mechanism sentence one describes loading. Mechanism sentence two describes space. Mechanism sentence three stays grounded.",
            "embed_text": "Article A Mechanism Mechanism sentence one describes loading.",
            "source_relpath": "source/a/index.html",
            "token_len": 21,
            "word_len": 12,
        },
        {
            "article_id": "article-b",
            "chunk_id": "chunk-b1",
            "title": "Article B",
            "section": "Main",
            "chunk_idx": "0",
            "article_seq": 0,
            "body": "Another source sentence explains shoulder motion. It has enough words for a compact evidence span.",
            "embed_text": "Article B Main Another source sentence explains shoulder motion.",
            "source_relpath": "source/b/index.html",
            "token_len": 22,
            "word_len": 13,
        },
    ]
    input_file = tmp_path / "chunks.parquet"
    out_dir = tmp_path / "hierarchical"
    pd.DataFrame(rows).to_parquet(input_file, index=False)

    first_manifest = build_hierarchical_corpus(input_file, out_dir)
    first_sections = read_jsonl(out_dir / "sections.jsonl")
    first_spans = read_jsonl(out_dir / "evidence_spans.jsonl")

    required_files = {
        "articles.jsonl",
        "sections.jsonl",
        "paragraphs.jsonl",
        "evidence_spans.jsonl",
        "corpus_manifest.json",
    }
    assert required_files <= {path.name for path in out_dir.iterdir()}

    articles = read_jsonl(out_dir / "articles.jsonl")
    sections = read_jsonl(out_dir / "sections.jsonl")
    spans = read_jsonl(out_dir / "evidence_spans.jsonl")

    assert len(articles) == 2
    assert first_manifest["article_count"] == 2
    assert len(articles) == pd.DataFrame(rows)["source_relpath"].nunique()

    article_ids = {article["article_id"] for article in articles}
    section_ids = {section["section_id"] for section in sections}
    assert all(section["article_id"] in article_ids for section in sections)
    assert all(span["article_id"] in article_ids for span in spans)
    assert all(span["section_id"] in section_ids for span in spans)

    build_hierarchical_corpus(input_file, out_dir)
    second_sections = read_jsonl(out_dir / "sections.jsonl")
    second_spans = read_jsonl(out_dir / "evidence_spans.jsonl")
    assert [row["section_id"] for row in first_sections] == [row["section_id"] for row in second_sections]
    assert [row["span_id"] for row in first_spans] == [row["span_id"] for row in second_spans]
