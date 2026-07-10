#!/usr/bin/env python3
"""
build_contextual_chunks.py — Contextual Retrieval augmentation for the MSK corpus.

Each chunk in ``chunks.parquet`` is embedded and BM25-indexed on its own text,
which strips away the surrounding article. A chunk that says "this muscle is
usually inhibited, not tight" never states *which* muscle or *which* syndrome,
so it embeds poorly against a user question that names them. This script writes a
short, LLM-generated *context prefix* that situates each chunk inside its article
before (re-)indexing — Anthropic's "Contextual Retrieval" technique.

It produces NEW side-by-side artifacts and never mutates chunks.parquet or the
existing chroma_store/. Adoption is gated on a gold-set eval comparison
(see scripts/rebuild_chroma_contextual.py and the eval harness).

Two-stage, cost-efficient design (corpus is only ~20 articles / ~1300 chunks):
  1. One compact *article summary* per article  (~20 utility-model calls).
  2. One *context prefix* per chunk, situated against that article summary plus
     its section heading                          (~1 utility-model call/chunk).

Outputs (under MSKArticlesINDEX/contextual/):
  - article_summaries.json       — {article_id: summary}
  - chunks_contextual.parquet    — chunks.parquet + two columns:
        context_prefix           — the generated situating sentence(s)
        contextual_embed_text     — prefix + "\n\n" + embed_text  (what to index)
  - build_manifest.json          — model, counts, token/cost totals, timestamp

Usage:
    # Zero-cost plan + cost estimate — ALWAYS run first.
    python scripts/build_contextual_chunks.py --dry-run

    # Bounded paid run (cost-guarded), e.g. first 50 chunks:
    python scripts/build_contextual_chunks.py --max-chunks 50 \
        --price-input-per-1k 0.0001 --price-output-per-1k 0.0004 \
        --max-estimated-cost-usd 1.00

    # Full paid build:
    python scripts/build_contextual_chunks.py \
        --price-input-per-1k 0.0001 --price-output-per-1k 0.0004 \
        --max-estimated-cost-usd 5.00

Requires: pip install openai pandas pyarrow python-dotenv tiktoken
          OPENAI_API_KEY in .env  (paid runs only; --dry-run needs no key)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional
    load_dotenv = None

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - heuristic fallback
    _ENC = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "MSKArticlesINDEX" / "chunks.parquet"
OUT_DIR = PROJECT_ROOT / "MSKArticlesINDEX" / "contextual"

UTILITY_MODEL = "gpt-4.1-nano"           # cheap situating model
ARTICLE_SUMMARY_MAX_TOKENS = 200
CONTEXT_PREFIX_MAX_TOKENS = 90           # keep prefixes short — they must not swamp the chunk
ARTICLE_TEXT_BUDGET_TOKENS = 6000        # cap article text sent for summarization

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("contextual")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


def truncate_tokens(text: str, max_tokens: int) -> str:
    if _ENC is None:
        return text[: max_tokens * 4]
    toks = _ENC.encode(text)
    if len(toks) <= max_tokens:
        return text
    return _ENC.decode(toks[:max_tokens])


# ── Prompt builders (pure — exercised in --dry-run without any API call) ───────

ARTICLE_SUMMARY_SYSTEM = (
    "You are a terse backend utility for an MSK biomechanics retrieval system. "
    "Follow the instruction exactly and return only the requested output."
)


def build_article_summary_prompt(title: str, article_text: str) -> str:
    body = truncate_tokens(article_text, ARTICLE_TEXT_BUDGET_TOKENS)
    return (
        "Summarize the following MSK biomechanics article in 3-4 sentences. "
        "State the main condition/region it covers, the key biomechanical "
        "mechanisms it argues for, and any corrective principles it emphasizes. "
        "Plain prose, no preamble.\n\n"
        f"Title: {title}\n\n"
        f"Article:\n{body}\n\n"
        "Summary:"
    )


def build_context_prefix_prompt(title: str, section: str, article_summary: str,
                                chunk_text: str) -> str:
    return (
        "You situate a text chunk within its article so a search engine can find it.\n"
        "Given the article context and the chunk, write ONE short sentence (max 30 words) "
        "that states what specific topic, structure, or mechanism this chunk is about and "
        "how it fits the article. Name the condition/region/muscles explicitly if implied. "
        "Do NOT summarize the whole article. Do NOT add facts not present. "
        "Return ONLY the sentence.\n\n"
        f"Article title: {title}\n"
        f"Article summary: {article_summary}\n"
        f"Section: {section or '(none)'}\n\n"
        f"Chunk:\n{chunk_text}\n\n"
        "Situating sentence:"
    )


# ── OpenAI call (only reached on paid runs) ────────────────────────────────────

def _make_client():
    from openai import OpenAI
    return OpenAI()


def _utility_call(client, prompt: str, system: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=UTILITY_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=max_tokens,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build contextual-retrieval augmented chunks.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan + estimate cost only. No API calls, no writes to artifacts.")
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="Process at most N chunks (bounded paid run). Default: all.")
    ap.add_argument("--price-input-per-1k", type=float, default=0.0,
                    help="USD per 1k input tokens for cost estimate/guard.")
    ap.add_argument("--price-output-per-1k", type=float, default=0.0,
                    help="USD per 1k output tokens for cost estimate/guard.")
    ap.add_argument("--max-estimated-cost-usd", type=float, default=None,
                    help="Abort a paid run if the pre-flight estimate exceeds this.")
    ap.add_argument("--chunks-path", type=Path, default=CHUNKS_PATH)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    if not args.chunks_path.exists():
        log.error("chunks parquet not found: %s", args.chunks_path)
        return 2

    df = pd.read_parquet(args.chunks_path)
    textcol = "embed_text" if "embed_text" in df.columns else "body"
    bodycol = "body" if "body" in df.columns else textcol
    log.info("Loaded %d chunks from %s (text col: %s)", len(df), args.chunks_path, textcol)

    if args.max_chunks is not None:
        df = df.head(args.max_chunks).copy()
        log.info("Limited to first %d chunks", len(df))

    # Group by article, preserving order.
    id_col = "article_id" if "article_id" in df.columns else "source_relpath"
    articles: Dict[str, pd.DataFrame] = {aid: g for aid, g in df.groupby(id_col, sort=False)}
    log.info("Spanning %d articles", len(articles))

    # ── Cost estimate (pre-flight) ────────────────────────────────────────────
    est_in = est_out = 0
    for aid, g in articles.items():
        title = str(g.iloc[0].get("title", ""))
        article_text = "\n\n".join(str(t) for t in g[bodycol].tolist())
        est_in += count_tokens(build_article_summary_prompt(title, article_text))
        est_out += ARTICLE_SUMMARY_MAX_TOKENS
        placeholder_summary = "x" * 400  # ~summary-sized stand-in for estimation
        for _, row in g.iterrows():
            est_in += count_tokens(build_context_prefix_prompt(
                title, str(row.get("section", "")), placeholder_summary, str(row[bodycol])))
            est_out += CONTEXT_PREFIX_MAX_TOKENS

    est_cost = (est_in / 1000.0) * args.price_input_per_1k + \
               (est_out / 1000.0) * args.price_output_per_1k
    log.info("Pre-flight estimate: %d articles + %d chunks | ~%d input tok, ~%d output tok | est cost $%.4f",
             len(articles), len(df), est_in, est_out, est_cost)

    if args.dry_run:
        # Show one example prompt so the plan is inspectable.
        first_aid = next(iter(articles))
        g = articles[first_aid]
        title = str(g.iloc[0].get("title", ""))
        example = build_context_prefix_prompt(
            title, str(g.iloc[0].get("section", "")), "<article summary here>",
            str(g.iloc[0][bodycol])[:400])
        log.info("DRY RUN — no API calls, no artifacts written.")
        print("\n----- EXAMPLE CONTEXT-PREFIX PROMPT -----\n" + example + "\n-----------------------------------------\n")
        return 0

    if args.max_estimated_cost_usd is not None and est_cost > args.max_estimated_cost_usd:
        log.error("Estimated cost $%.4f exceeds --max-estimated-cost-usd $%.4f; aborting.",
                  est_cost, args.max_estimated_cost_usd)
        return 3

    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")
    client = _make_client()

    # ── Stage 1: article summaries ────────────────────────────────────────────
    summaries: Dict[str, str] = {}
    for i, (aid, g) in enumerate(articles.items(), 1):
        title = str(g.iloc[0].get("title", ""))
        article_text = "\n\n".join(str(t) for t in g[bodycol].tolist())
        prompt = build_article_summary_prompt(title, article_text)
        try:
            summaries[aid] = _utility_call(client, prompt, ARTICLE_SUMMARY_SYSTEM,
                                           ARTICLE_SUMMARY_MAX_TOKENS)
        except Exception as exc:
            log.warning("article summary failed for %s (%s); using title fallback", aid, exc)
            summaries[aid] = title
        log.info("  article summary %d/%d", i, len(articles))

    # ── Stage 2: per-chunk context prefixes ───────────────────────────────────
    prefixes: List[str] = []
    contextual_text: List[str] = []
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        aid = row[id_col]
        title = str(row.get("title", ""))
        section = str(row.get("section", ""))
        summary = summaries.get(aid, title)
        chunk_text = str(row[bodycol])
        prompt = build_context_prefix_prompt(title, section, summary, chunk_text)
        try:
            prefix = _utility_call(client, prompt, ARTICLE_SUMMARY_SYSTEM,
                                   CONTEXT_PREFIX_MAX_TOKENS)
        except Exception as exc:
            log.warning("context prefix failed for chunk %s (%s); using title+section", i, exc)
            prefix = f"{title} — {section}".strip(" —")
        prefix = " ".join(prefix.split())
        prefixes.append(prefix)
        # Index text = prefix + original embed_text (which already has title/section).
        contextual_text.append(f"{prefix}\n\n{row[textcol]}")
        if i % 50 == 0 or i == n:
            log.info("  context prefix %d/%d", i, n)

    df = df.copy()
    df["context_prefix"] = prefixes
    df["contextual_embed_text"] = contextual_text

    # ── Write artifacts ───────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "article_summaries.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    out_parquet = args.out_dir / "chunks_contextual.parquet"
    df.to_parquet(out_parquet, index=False)
    manifest = {
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "utility_model": UTILITY_MODEL,
        "n_articles": len(articles),
        "n_chunks": int(len(df)),
        "estimated_input_tokens": int(est_in),
        "estimated_output_tokens": int(est_out),
        "estimated_cost_usd": round(est_cost, 6),
        "source_chunks": str(args.chunks_path),
        "text_col": textcol,
    }
    (args.out_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    log.info("Wrote %s (%d chunks)", out_parquet, len(df))
    log.info("Wrote article_summaries.json and build_manifest.json to %s", args.out_dir)
    log.info("Next: python scripts/rebuild_chroma_contextual.py  (builds side-by-side store)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
