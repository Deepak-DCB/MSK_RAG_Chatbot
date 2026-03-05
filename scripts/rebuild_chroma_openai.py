#!/usr/bin/env python3
"""
rebuild_chroma_openai.py — One-time script to rebuild chroma_store
using OpenAI text-embedding-3-small (1536-dim) embeddings.

Usage:
    python scripts/rebuild_chroma_openai.py

Requires:
    pip install openai chromadb pandas pyarrow python-dotenv numpy
    OPENAI_API_KEY in .env
"""

import os
import re
import time
import logging
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "MSKArticlesINDEX" / "chunks.parquet"
STORE_DIR = PROJECT_ROOT / "chroma_store"
COLLECTION_NAME = "msk_chunks"
EMBED_MODEL = "text-embedding-3-large"  # 3072-dim, higher retrieval accuracy
BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs per call
MIN_LEN = 50
CAPTION_RE = re.compile(r"\b(fig(?:ure)?|source|click|image|photo|credit)\b", re.I)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("rebuild")

# ── Helpers (mirrored from ChromaDB.py) ───────────────────────────────────────

def is_bad(t: str) -> bool:
    """Reject empty, short, or caption-like text."""
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


def make_metadata(df: pd.DataFrame, textcol: str) -> List[Dict[str, Any]]:
    """Convert rows to serializable metadata dicts (minus text column)."""
    mdf = df.drop(columns=[textcol], errors="ignore")
    records = []
    for row in mdf.to_dict(orient="records"):
        clean = {k: norm_val(v) for k, v in row.items()}
        records.append({k: v for k, v in clean.items() if v is not None})
    return records


def embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Call OpenAI embeddings API for a batch of texts."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    openai_client = OpenAI(api_key=api_key)

    # 1) Load chunks
    log.info("📂 Loading %s", CHUNKS_PATH)
    df = pd.read_parquet(CHUNKS_PATH)
    log.info("   %d rows loaded", len(df))

    # 2) Pick text column
    textcol = "embed_text" if "embed_text" in df.columns else "body"
    log.info("📝 Using text column: %s", textcol)

    # 3) Filter bad chunks
    mask = ~df[textcol].fillna("").apply(is_bad)
    df = df[mask].reset_index(drop=True)
    log.info("📉 After filtering: %d chunks remain", len(df))

    # 4) Prepare texts
    texts = df[textcol].astype(str).tolist()

    # 5) Batch-embed with OpenAI
    all_embeddings: List[List[float]] = []
    total = len(texts)
    log.info("🚀 Embedding %d chunks with %s (batch=%d)…", total, EMBED_MODEL, BATCH_SIZE)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = texts[start:end]
        try:
            embs = embed_batch(openai_client, batch)
            all_embeddings.extend(embs)
            log.info("   → %d / %d", end, total)
        except Exception as e:
            log.error("❌ Batch %d–%d failed: %s", start, end, e)
            log.info("   Retrying in 5s…")
            time.sleep(5)
            embs = embed_batch(openai_client, batch)
            all_embeddings.extend(embs)
            log.info("   → %d / %d (retry OK)", end, total)

    log.info("✅ Got %d embeddings, dim=%d", len(all_embeddings), len(all_embeddings[0]))

    # 6) Build Chroma store
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(STORE_DIR))

    # Overwrite existing collection
    try:
        client.delete_collection(COLLECTION_NAME)
        log.info("♻️  Deleted existing collection '%s'", COLLECTION_NAME)
    except Exception:
        pass

    coll = client.get_or_create_collection(COLLECTION_NAME)

    ids = (
        df["chunk_id"].astype(str).tolist()
        if "chunk_id" in df.columns
        else [str(i) for i in range(len(df))]
    )
    docs = texts
    metas = make_metadata(df, textcol)

    insert_batch = 500
    log.info("🧩 Inserting %d chunks into Chroma…", len(ids))
    for s in range(0, len(ids), insert_batch):
        e = min(s + insert_batch, len(ids))
        coll.add(
            ids=ids[s:e],
            documents=docs[s:e],
            metadatas=metas[s:e],
            embeddings=all_embeddings[s:e],
        )
        log.info("   → %d / %d", e, len(ids))

    log.info("🔎 Final count: %d", coll.count())
    log.info("✅ Done! chroma_store written to: %s", STORE_DIR)

    # 7) Record the embedding model used
    model_file = PROJECT_ROOT / "embeddings" / "embedding_model.txt"
    model_file.write_text(EMBED_MODEL, encoding="utf-8")
    log.info("📝 Updated %s → %s", model_file, EMBED_MODEL)


if __name__ == "__main__":
    main()
