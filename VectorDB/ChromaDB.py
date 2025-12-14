#!/usr/bin/env python3
"""
chromadb.py (v4.6 — stable persistent Chroma builder, Chroma 1.3.x API)

Builds a *persistent* Chroma vector store from chunks.parquet + embeddings.npy
with clean metadata and no deprecated calls.

Changes from v4.5
─────────────────
✓ Uses chromadb.PersistentClient(path=...) instead of chromadb.Client()
✓ Removed redundant CHROMA_DATA_PATH env var (ignored by new API)
✓ Verified batch-safe insertion and overwrite behavior
✓ Improved logging clarity for persistence and final count

Usage:
    python chromadb.py --chunks ./MSKArticlesINDEX/chunks.parquet \
                       --embeddings ./embeddings/embeddings.npy \
                       --persist-dir ./chroma_store \
                       --name msk_chunks \
                       --overwrite
"""

from __future__ import annotations
import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Logging & constants
# ──────────────────────────────────────────────────────────────────────────────
log = logging.getLogger("chromadb_builder")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS = PROJECT_ROOT / "MSKArticlesINDEX" / "chunks.parquet"
DEFAULT_EMBEDS = PROJECT_ROOT / "embeddings" / "embeddings.npy"
DEFAULT_STORE = PROJECT_ROOT / "chroma_store"
DEFAULT_COLLECTION = "msk_chunks"
DEFAULT_BATCH = 500
MIN_LEN = 50
CAPTION_RE = re.compile(r"\b(fig(?:ure)?|source|click|image|photo|credit)\b", re.I)

# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────
def load(chunks: Path, embeds: Path):
    log.info("📂 Loading chunks: %s", chunks)
    df = pd.read_parquet(chunks)
    log.info("📂 Loading embeddings: %s", embeds)
    embs = np.load(embeds)
    if len(df) != embs.shape[0]:
        raise ValueError(f"Row mismatch: chunks={len(df)} vs embs={embs.shape[0]}")
    log.info("✅ Loaded %d chunks (%d-dim vectors)", len(df), embs.shape[1])
    return df, embs


def pick_text(df, prefer_images: bool) -> str:
    if prefer_images and "text_with_images" in df.columns:
        log.info("📝 Using 'text_with_images'")
        return "text_with_images"
    if "embed_text" in df.columns:
        log.info("📝 Using 'embed_text'")
        return "embed_text"
    raise ValueError("No usable text column found (expected 'embed_text' or 'text_with_images').")


def is_bad(t: str) -> bool:
    """Reject empty, short, or caption-like text."""
    return (not t) or len(t) < MIN_LEN or bool(CAPTION_RE.search(t))


def filter_df(df: pd.DataFrame, embs: np.ndarray, textcol: str):
    mask = ~df[textcol].fillna("").apply(is_bad)
    out_df = df[mask].reset_index(drop=True)
    out_embs = embs[mask.to_numpy()]
    log.info("📉 Filtered %d invalid chunks → %d remain", len(df) - len(out_df), len(out_df))
    return out_df, out_embs


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


def meta(df: pd.DataFrame, textcol: str) -> List[Dict[str, Any]]:
    """Convert DataFrame rows (minus text) to serializable metadata dicts."""
    mdf = df.drop(columns=[textcol], errors="ignore")
    records = []
    for row in mdf.to_dict(orient="records"):
        clean = {k: norm_val(v) for k, v in row.items()}
        records.append({k: v for k, v in clean.items() if v is not None})
    return records

# ──────────────────────────────────────────────────────────────────────────────
# Main build routine
# ──────────────────────────────────────────────────────────────────────────────
def build(chunks, embeds, store, name, use_images, overwrite, batch):
    store.mkdir(parents=True, exist_ok=True)
    df, embs = load(chunks, embeds)
    textcol = pick_text(df, use_images)
    df, embs = filter_df(df, embs, textcol)

    log.info("🗄️ Initializing persistent Chroma client at: %s", store)
    client = chromadb.PersistentClient(path=str(store))

    # optional overwrite
    if overwrite:
        try:
            client.delete_collection(name)
            log.info("♻️ Overwrote existing collection '%s'", name)
        except Exception:
            log.warning("No existing collection to overwrite, continuing...")

    coll = client.get_or_create_collection(name)

    # Prepare data
    ids = (
        df["chunk_id"].astype(str).tolist()
        if "chunk_id" in df.columns
        else [str(i) for i in range(len(df))]
    )
    docs = df[textcol].astype(str).tolist()
    metas = meta(df, textcol)

    # Insert in batches
    log.info("🧩 Inserting %d chunks in batches of %d…", len(ids), batch)
    for s in range(0, len(ids), batch):
        e = min(s + batch, len(ids))
        coll.add(
            ids=ids[s:e],
            documents=docs[s:e],
            metadatas=metas[s:e],
            embeddings=embs[s:e].tolist(),
        )
        log.info("  → %d / %d", e, len(ids))

    # Verify persistence
    final_count = coll.count()
    log.info("🔎 Final count in collection '%s': %d", name, final_count)
    log.info("✅ Done. Persistent Chroma store written to: %s", store)
    return final_count

# ──────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Rebuild persistent Chroma store (stable API)")
    p.add_argument("-c", "--chunks", type=Path, default=DEFAULT_CHUNKS)
    p.add_argument("-e", "--embeddings", type=Path, default=DEFAULT_EMBEDS)
    p.add_argument("-d", "--persist-dir", type=Path, default=DEFAULT_STORE)
    p.add_argument("-n", "--name", type=str, default=DEFAULT_COLLECTION)
    p.add_argument("--use-images", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    args = p.parse_args()

    build(
        args.chunks,
        args.embeddings,
        args.persist_dir,
        args.name,
        args.use_images,
        args.overwrite,
        args.batch_size,
    )

if __name__ == "__main__":
    main()
