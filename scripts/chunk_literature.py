#!/usr/bin/env python3
"""
chunk_literature.py — Step 4a of the cited-literature backfill.

Chunks literature_corpus.jsonl into the SAME chunk schema the blog corpus uses,
by reusing textExtract's own chunking primitives (identical sentence grouping,
windowing, and 512-token hard cap). Adds the provenance/tiering columns decided
for ingest:

  source_type   = "literature"
  evidence_tier = "peer_reviewed"   (all works here are DOI/PMID-resolved)
  doi, year, venue, authors, url, cited_by_n

Output (default under MSKArticlesINDEX/):
  literature_chunks.parquet   base chunk schema + provenance columns

This does NOT touch chroma_store/. rebuild_chroma_openai.py --with-literature
merges this parquet, backfills the blog tier, and indexes both (that step calls
the embeddings API — run it deliberately).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from Text_Extraction import textExtract as TE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=PROJECT_ROOT / "MSKArticlesINDEX" / "literature_corpus.jsonl")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "MSKArticlesINDEX" / "literature_chunks.parquet")
    args = ap.parse_args()

    cfg = TE.Config()                 # same defaults as the blog extractor
    tok = TE.TokenCounter(cfg)

    works = [json.loads(l) for l in args.input.open(encoding="utf-8")]
    rows = []
    n_with_text = 0
    for w in works:
        text = w.get("text")
        if not text:
            continue
        n_with_text += 1
        # Stable article_id from the strongest identifier available.
        ident = w.get("openalex_id") or w.get("doi") or w.get("title") or ""
        article_id = TE.sha256_hex(str(ident))
        section = "Full text" if w.get("text_source") == "oa_fulltext" else "Abstract"
        source_relpath = w.get("doi") or w.get("openalex_id") or article_id

        chunk_rows, _ = TE._materialize_chunks_for_block(
            block_text=text,
            header_title=w.get("title") or "",
            section=section,
            cfg=cfg,
            tok=tok,
            img_texts=[],
            images_for_block=[],
            article_id=article_id,
            article_seq_start=0,
            source_relpath=source_relpath,
        )
        authors = w.get("authors") or []
        cited_by = w.get("cited_by") or []
        for r in chunk_rows:
            r.update({
                "source_type": "literature",
                "evidence_tier": "peer_reviewed",
                "doi": w.get("doi"),
                "pmid": w.get("pmid"),
                "year": w.get("year"),
                "venue": w.get("venue"),
                "authors": ", ".join(authors) if authors else None,
                "url": w.get("oa_url") or (f"https://doi.org/{w['doi']}" if w.get("doi") else None),
                "text_source": w.get("text_source"),
                "cited_by_n": len(cited_by),
            })
            rows.append(r)

    if not rows:
        print("No literature chunks produced (no works had text).")
        return

    df = pd.DataFrame(rows)
    # Same near-exact dedup the blog build uses.
    df = TE.dedup_near_exact(df, "body")
    df = df.sort_values(["source_relpath", "article_seq"]).reset_index(drop=True)
    df.to_parquet(args.out, index=False)

    print(f"Works with text        : {n_with_text}")
    print(f"Literature chunks      : {len(df)}")
    print(f"  from full text       : {(df['section'] == 'Full text').sum()}")
    print(f"  from abstracts       : {(df['section'] == 'Abstract').sum()}")
    print(f"Avg tokens/chunk       : {df['token_len'].mean():.0f} (max {df['token_len'].max()})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
