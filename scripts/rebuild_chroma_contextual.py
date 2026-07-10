#!/usr/bin/env python3
"""
rebuild_chroma_contextual.py — Build a SIDE-BY-SIDE Chroma store from the
contextual-retrieval augmented chunks produced by build_contextual_chunks.py.

This is a mirror of rebuild_chroma_openai.py with two deliberate differences:
  - it embeds ``contextual_embed_text`` (context prefix + original text), and
  - it writes to a SEPARATE store dir / collection so the production
    chroma_store/ is never touched.

The result is compared against the baseline via the eval harness before anything
adopts it (retrieval change discipline: build side-by-side, keep off by default,
gate adoption on gold-set metrics).

Point retrieval at it by setting, before starting the backend/eval:
    MSK_CHROMA_DIR   = <repo>/chroma_store_contextual
    MSK_COLLECTION   = msk_chunks_contextual
(qaEngine reads these env overrides; unset = production defaults.)

Usage:
    python scripts/rebuild_chroma_contextual.py --dry-run     # plan only, no API
    python scripts/rebuild_chroma_contextual.py               # full build (paid)

Requires: pip install openai chromadb pandas pyarrow python-dotenv numpy
          OPENAI_API_KEY in .env
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTEXTUAL_PARQUET = PROJECT_ROOT / "MSKArticlesINDEX" / "contextual" / "chunks_contextual.parquet"
STORE_DIR = PROJECT_ROOT / "chroma_store_contextual"
COLLECTION_NAME = "msk_chunks_contextual"
EMBED_MODEL = "text-embedding-3-large"   # MUST match the production embedder
BATCH_SIZE = 100
MIN_LEN = 50
CAPTION_RE = re.compile(r"\b(fig(?:ure)?|source|click|image|photo|credit)\b", re.I)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("rebuild-contextual")


def is_bad(t: str) -> bool:
    return (not t) or len(t) < MIN_LEN or bool(CAPTION_RE.search(t))


def norm_val(v: Any):
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return ", ".join(map(str, v))
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def make_metadata(df: pd.DataFrame, drop_cols: List[str]) -> List[Dict[str, Any]]:
    mdf = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    records = []
    for row in mdf.to_dict(orient="records"):
        clean = {k: norm_val(v) for k, v in row.items()}
        records.append({k: v for k, v in clean.items() if v is not None})
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Plan only; no API calls or writes.")
    ap.add_argument("--parquet", type=Path, default=CONTEXTUAL_PARQUET)
    ap.add_argument("--store-dir", type=Path, default=STORE_DIR)
    args = ap.parse_args(argv)

    if not args.parquet.exists():
        log.error("Contextual parquet not found: %s", args.parquet)
        log.error("Run scripts/build_contextual_chunks.py first.")
        return 2

    df = pd.read_parquet(args.parquet)
    if "contextual_embed_text" not in df.columns:
        log.error("Column 'contextual_embed_text' missing — was this built by build_contextual_chunks.py?")
        return 2

    textcol = "contextual_embed_text"
    # Filter on the ORIGINAL body so we drop the same caption/short chunks as baseline.
    filter_col = "body" if "body" in df.columns else textcol
    mask = ~df[filter_col].fillna("").apply(is_bad)
    df = df[mask].reset_index(drop=True)
    log.info("After filtering: %d chunks", len(df))

    texts = df[textcol].astype(str).tolist()

    if args.dry_run:
        log.info("DRY RUN — would embed %d contextual chunks with %s into %s (collection '%s').",
                 len(texts), EMBED_MODEL, args.store_dir, COLLECTION_NAME)
        log.info("Example indexed text (first 300 chars):\n%s", texts[0][:300] if texts else "(none)")
        return 0

    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set")
        return 2

    import chromadb
    from openai import OpenAI

    client_ai = OpenAI()

    def embed_batch(batch: List[str]) -> List[List[float]]:
        resp = client_ai.embeddings.create(model=EMBED_MODEL, input=batch)
        return [d.embedding for d in resp.data]

    all_emb: List[List[float]] = []
    total = len(texts)
    log.info("Embedding %d chunks (batch=%d)…", total, BATCH_SIZE)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = texts[start:end]
        try:
            all_emb.extend(embed_batch(batch))
        except Exception as e:
            log.error("Batch %d-%d failed: %s; retrying in 5s", start, end, e)
            time.sleep(5)
            all_emb.extend(embed_batch(batch))
        log.info("  → %d / %d", end, total)

    args.store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.store_dir))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    coll = client.get_or_create_collection(COLLECTION_NAME)

    ids = df["chunk_id"].astype(str).tolist() if "chunk_id" in df.columns else [str(i) for i in range(len(df))]
    metas = make_metadata(df, drop_cols=[textcol])

    insert_batch = 500
    for s in range(0, len(ids), insert_batch):
        e = min(s + insert_batch, len(ids))
        coll.add(ids=ids[s:e], documents=texts[s:e], metadatas=metas[s:e], embeddings=all_emb[s:e])
        log.info("  inserted %d / %d", e, len(ids))

    log.info("Final count: %d in %s (collection '%s')", coll.count(), args.store_dir, COLLECTION_NAME)
    log.info("To eval against it: set MSK_CHROMA_DIR=%s and MSK_COLLECTION=%s", args.store_dir, COLLECTION_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
