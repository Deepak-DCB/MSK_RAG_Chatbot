#!/usr/bin/env python3
"""Build hierarchical corpus artifacts from existing MSK chunk data.

This phase intentionally supports the currently committed artifact only:
``MSKArticlesINDEX/chunks.parquet``. If raw HTML mirrors are restored later,
the reconstruction boundary is isolated in ``build_from_chunks`` so a raw-source
builder can be added without changing downstream artifact schemas.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency fallback
    tiktoken = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "MSKArticlesINDEX" / "chunks.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "MSKArticlesINDEX" / "hierarchical"
DEFAULT_EMBEDDING_MODEL_FILE = PROJECT_ROOT / "embeddings" / "embedding_model.txt"
SCHEMA_VERSION = "1.0.0"
RECONSTRUCTION_METHOD = "from_chunks"


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_hash(*parts: Any, length: int = 32) -> str:
    payload = "|".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("o200k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return int(round(len(text.split()) * 1.33))


def word_count(text: str) -> int:
    return len(str(text or "").split())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_ws(text).encode("utf-8")).hexdigest()


def read_embedding_model(path: Path = DEFAULT_EMBEDDING_MODEL_FILE) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip() or "unknown"
    return "unknown"


def dedupe_append(existing: str, addition: str) -> str:
    """Append text while suppressing common chunk overlap.

    The source chunks can overlap by a few sentences. This heuristic removes an
    exact word suffix/prefix overlap up to 80 words while staying deterministic.
    """
    existing = normalize_ws(existing)
    addition = normalize_ws(addition)
    if not existing:
        return addition
    if not addition:
        return existing

    existing_words = existing.split()
    add_words = addition.split()
    max_overlap = min(80, len(existing_words), len(add_words))
    for n in range(max_overlap, 8, -1):
        if existing_words[-n:] == add_words[:n]:
            return " ".join(existing_words + add_words[n:])
    return f"{existing}\n\n{addition}"


def split_sentences(text: str) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text) if s.strip()]


def group_evidence_spans(text: str, max_tokens: int = 180) -> List[str]:
    sentences = split_sentences(text)
    spans: List[str] = []
    buf: List[str] = []

    def flush() -> None:
        nonlocal buf
        span = normalize_ws(" ".join(buf))
        if word_count(span) >= 12:
            spans.append(span)
        buf = []

    for sentence in sentences:
        candidate = normalize_ws(" ".join(buf + [sentence]))
        if buf and (len(buf) >= 3 or count_tokens(candidate) > max_tokens):
            flush()
        if count_tokens(sentence) > max_tokens:
            words = sentence.split()
            for start in range(0, len(words), 110):
                piece = " ".join(words[start:start + 110])
                if word_count(piece) >= 12:
                    spans.append(piece)
            continue
        buf.append(sentence)
    if buf:
        flush()
    return spans


def jsonl_write(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_from_chunks(df: pd.DataFrame, created_at: str, source_hash: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    required = {"source_relpath", "article_seq", "body", "section", "chunk_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"chunks.parquet missing required columns: {', '.join(missing)}")

    articles: List[Dict[str, Any]] = []
    sections: List[Dict[str, Any]] = []
    paragraphs: List[Dict[str, Any]] = []
    spans: List[Dict[str, Any]] = []

    df = df.copy()
    df["source_relpath"] = df["source_relpath"].astype(str)
    df["article_seq"] = pd.to_numeric(df["article_seq"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["source_relpath", "article_seq", "chunk_idx" if "chunk_idx" in df.columns else "chunk_id"])

    for source_order, (source_relpath, article_df) in enumerate(df.groupby("source_relpath", sort=True)):
        article_df = article_df.sort_values("article_seq")
        first = article_df.iloc[0]
        title = normalize_ws(first.get("title") or source_relpath)
        article_id = str(first.get("article_id") or stable_hash(source_relpath))

        section_ids: List[str] = []
        article_text = ""

        for section_order, (section_name_raw, sec_df) in enumerate(article_df.groupby("section", sort=False)):
            section_name = normalize_ws(section_name_raw or "Main") or "Main"
            sec_df = sec_df.sort_values("article_seq")
            section_text = ""
            chunk_ids: List[str] = []

            for paragraph_order, (_, chunk_row) in enumerate(sec_df.iterrows()):
                body = normalize_ws(chunk_row.get("body") or chunk_row.get("embed_text") or "")
                if not body:
                    continue
                chunk_id = str(chunk_row.get("chunk_id") or stable_hash(article_id, section_name, paragraph_order, body[:80]))
                chunk_ids.append(chunk_id)
                paragraph_id = stable_hash("paragraph", article_id, section_name, paragraph_order, chunk_id)
                paragraphs.append({
                    "paragraph_id": paragraph_id,
                    "article_id": article_id,
                    "title": title,
                    "source_relpath": source_relpath,
                    "section_name": section_name,
                    "paragraph_order": paragraph_order,
                    "text": body,
                    "token_len": count_tokens(body),
                    "word_len": word_count(body),
                    "source_chunk_id": chunk_id,
                    "schema_version": SCHEMA_VERSION,
                })
                section_text = dedupe_append(section_text, body)

            if not section_text:
                continue

            section_id = stable_hash("section", article_id, section_order, section_name)
            section_ids.append(section_id)
            sections.append({
                "section_id": section_id,
                "article_id": article_id,
                "title": title,
                "source_relpath": source_relpath,
                "section_name": section_name,
                "section_order": section_order,
                "text": section_text,
                "token_len": count_tokens(section_text),
                "word_len": word_count(section_text),
                "chunk_ids": chunk_ids,
                "source_hash": text_hash(section_text),
                "schema_version": SCHEMA_VERSION,
            })

            for span_order, span_text in enumerate(group_evidence_spans(section_text)):
                spans.append({
                    "span_id": stable_hash("span", section_id, span_order, span_text[:120]),
                    "article_id": article_id,
                    "section_id": section_id,
                    "source_relpath": source_relpath,
                    "title": title,
                    "section_name": section_name,
                    "span_order": span_order,
                    "text": span_text,
                    "token_len": count_tokens(span_text),
                    "word_len": word_count(span_text),
                    "source_chunk_ids": chunk_ids,
                    "schema_version": SCHEMA_VERSION,
                })

            article_text = dedupe_append(article_text, section_text)

        articles.append({
            "article_id": article_id,
            "title": title,
            "source_relpath": source_relpath,
            "reconstructed_text": article_text,
            "section_ids": section_ids,
            "token_len": count_tokens(article_text),
            "word_len": word_count(article_text),
            "reconstruction_method": RECONSTRUCTION_METHOD,
            "source_hash": text_hash(article_text),
            "created_at": created_at,
            "schema_version": SCHEMA_VERSION,
        })

    return articles, sections, paragraphs, spans


def build_hierarchical_corpus(input_file: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input chunks file not found: {input_file}")

    created_at = utc_now_iso()
    source_hash = file_sha256(input_file)
    df = pd.read_parquet(input_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    articles, sections, paragraphs, spans = build_from_chunks(df, created_at, source_hash)

    jsonl_write(output_dir / "articles.jsonl", articles)
    jsonl_write(output_dir / "sections.jsonl", sections)
    jsonl_write(output_dir / "paragraphs.jsonl", paragraphs)
    jsonl_write(output_dir / "evidence_spans.jsonl", spans)

    warnings = []
    if len(articles) != int(df["source_relpath"].nunique()):
        warnings.append("Article count differs from unique source_relpath count.")
    if paragraphs:
        warnings.append("Paragraphs were reconstructed from source chunks, not raw HTML paragraph boundaries.")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": created_at,
        "input_file": str(input_file),
        "input_sha256": source_hash,
        "input_row_count": int(len(df)),
        "article_count": len(articles),
        "section_count": len(sections),
        "paragraph_count": len(paragraphs),
        "span_count": len(spans),
        "embedding_model": read_embedding_model(),
        "reconstruction_method": RECONSTRUCTION_METHOD,
        "notes": [
            "Built from chunks.parquet because raw HTML source is optional for this phase.",
            "Full article and section text is reconstructed from ordered chunk bodies with exact overlap suppression.",
        ],
        "warnings": warnings,
    }
    (output_dir / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hierarchical MSK corpus artifacts from chunks.parquet")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_hierarchical_corpus(args.input, args.output_dir)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "article_count": manifest["article_count"],
        "section_count": manifest["section_count"],
        "span_count": manifest["span_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
