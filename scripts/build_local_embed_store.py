#!/usr/bin/env python3
"""
build_local_embed_store.py — build a Chroma store with LOCAL SentenceTransformer
embeddings, so the app can run a fully non-OpenAI pipeline (no OpenAI key at all:
local embeddings -> this store -> BM25 -> a free generation provider like Groq).

It is the local-embedding twin of scripts/rebuild_chroma_openai.py:
  * same chunk source (MSKArticlesINDEX/chunks.parquet) and same bad-chunk filter,
    so the corpus matches the production store one-for-one;
  * embeds with a local SentenceTransformer, L2-normalized for cosine — the SAME
    normalization qaEngine.local_embed() applies at query time, so query and
    document vectors share one space;
  * writes to a SEPARATE store dir + collection (default chroma_store_local /
    msk_chunks) so production chroma_store is never touched.

To serve it, point the app at the local backend + this store:
    MSK_EMBED_PROVIDER=local
    MSK_LOCAL_EMBED_MODEL=mixedbread-ai/mxbai-embed-large-v1   # must match --model
    MSK_CHROMA_DIR=<abs path to chroma_store_local>
    MSK_COLLECTION=msk_chunks

Usage:
    python scripts/build_local_embed_store.py --dry-run
    python scripts/build_local_embed_store.py            # actually build
    python scripts/build_local_embed_store.py --model BAAI/bge-large-en-v1.5

Requires (dev extras, already in root requirements.txt):
    pip install sentence-transformers torch chromadb pandas pyarrow numpy
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "MSKArticlesINDEX" / "chunks.parquet"
DEFAULT_STORE_DIR = PROJECT_ROOT / "chroma_store_local"
DEFAULT_COLLECTION = "msk_chunks"
DEFAULT_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
BATCH_SIZE = 64
MIN_LEN = 50
CAPTION_RE = re.compile(r"\b(fig(?:ure)?|source|click|image|photo|credit)\b", re.I)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_local")


# ── Helpers (mirrored from rebuild_chroma_openai.py so the corpus matches) ─────

def is_bad(t: str) -> bool:
    """Reject empty, short, or caption-like text — identical filter to production."""
    return (not t) or len(t) < MIN_LEN or bool(CAPTION_RE.search(t))


def norm_val(v: Any):
    import numpy as np
    import pandas as pd
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return ", ".join(map(str, v))
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


def make_metadata(df, textcol: str) -> List[Dict[str, Any]]:
    mdf = df.drop(columns=[textcol], errors="ignore")
    records = []
    for row in mdf.to_dict(orient="records"):
        clean = {k: norm_val(v) for k, v in row.items()}
        records.append({k: v for k, v in clean.items() if v is not None})
    return records


def load_texts(textcol_out: List[str]):
    """Load + filter chunks; return (df, texts). textcol_out[0] receives the column."""
    import pandas as pd
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"{CHUNKS_PATH} not found — nothing to embed.")
    df = pd.read_parquet(CHUNKS_PATH)
    log.info("Loaded %d rows from %s", len(df), CHUNKS_PATH.name)
    textcol = "embed_text" if "embed_text" in df.columns else "body"
    textcol_out.append(textcol)
    mask = ~df[textcol].fillna("").apply(is_bad)
    df = df[mask].reset_index(drop=True)
    log.info("After filtering: %d chunks (text column: %s)", len(df), textcol)
    return df, df[textcol].astype(str).tolist()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model id")
    p.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--dry-run", action="store_true",
                   help="Load + filter chunks and report, but download/embed nothing.")
    args = p.parse_args(argv)

    textcol_out: List[str] = []
    df, texts = load_texts(textcol_out)

    if args.dry_run:
        log.info("DRY RUN — would embed %d chunks with '%s' (normalized) into %s "
                 "(collection '%s'). No model download, no writes.",
                 len(texts), args.model, args.store_dir, args.collection)
        return 0

    # Heavy imports only past the dry-run gate.
    import numpy as np
    import chromadb
    from sentence_transformers import SentenceTransformer

    log.info("Loading local model '%s'…", args.model)
    model = SentenceTransformer(args.model)

    log.info("Embedding %d chunks (batch=%d, L2-normalized)…", len(texts), args.batch_size)
    embs = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embs = np.asarray(embs, dtype=float)
    log.info("Got %d embeddings, dim=%d", len(embs), embs.shape[1] if len(embs) else 0)

    args.store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.store_dir))
    try:
        client.delete_collection(args.collection)
        log.info("Deleted existing collection '%s'", args.collection)
    except Exception:
        pass
    coll = client.get_or_create_collection(args.collection)

    ids = (df["chunk_id"].astype(str).tolist()
           if "chunk_id" in df.columns else [str(i) for i in range(len(df))])
    metas = make_metadata(df, textcol_out[0])

    insert_batch = 500
    for s in range(0, len(ids), insert_batch):
        e = min(s + insert_batch, len(ids))
        coll.add(ids=ids[s:e], documents=texts[s:e], metadatas=metas[s:e],
                 embeddings=[v.tolist() for v in embs[s:e]])
        log.info("  inserted %d / %d", e, len(ids))

    log.info("Final count: %d in %s (collection '%s')", coll.count(), args.store_dir, args.collection)

    # Record which model built this store, next to the OpenAI one's marker file.
    marker = PROJECT_ROOT / "embeddings" / "embedding_model_local.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(args.model, encoding="utf-8")
    log.info("Wrote model marker %s → %s", marker, args.model)
    log.info("To serve: MSK_EMBED_PROVIDER=local MSK_LOCAL_EMBED_MODEL=%s "
             "MSK_CHROMA_DIR=%s MSK_COLLECTION=%s", args.model, args.store_dir, args.collection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
