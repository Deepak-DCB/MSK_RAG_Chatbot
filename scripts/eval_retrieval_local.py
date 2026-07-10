#!/usr/bin/env python3
"""
eval_retrieval_local.py — key-free retrieval eval.

Question it answers: does the configured embedding backend actually *retrieve the
right chunks*? It scores Hit@k and MRR against datasets/retrieval-goldens.jsonl
(ground-truth chunk_ids), running the real hybrid_search() from qaEngine.

It also runs a **BM25-only baseline** on the same store, so the *added value* of the
embeddings is visible — not just absolute recall. If local-hybrid barely beats
BM25-only, the local embeddings aren't pulling their weight.

Runs with NO OpenAI key when the local backend is selected:

    MSK_EMBED_PROVIDER=local \
    MSK_CHROMA_DIR=<abs path to chroma_store_local> \
    MSK_COLLECTION=msk_chunks \
    python scripts/eval_retrieval_local.py --ks 5 10 20

Caveat: the shipped gold set is tiny (a few queries) — treat results as a smoke
signal, not a statistically settled number. Expand datasets/retrieval-goldens.jsonl
for a real verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "VectorDB"))

DEFAULT_GOLDENS = PROJECT_ROOT / "datasets" / "retrieval-goldens.jsonl"


def load_goldens(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def ranked_chunk_ids(raw: Dict[str, Any]) -> List[str]:
    """Extract retrieved chunk_ids (in rank order) from a hybrid_search result."""
    metas = (raw.get("metadatas") or [[]])[0]
    return [str((m or {}).get("chunk_id") or "") for m in metas]


def eval_mode(qa, goldens: List[Dict[str, Any]], ks: List[int], *, bm25_only: bool) -> List[Dict[str, Any]]:
    collection = qa._backend.load_collection()
    pool = max(50, max(ks))

    orig_encode = qa.encode_query
    if bm25_only:
        def _no_dense(_text):
            raise RuntimeError("dense disabled (BM25-only baseline)")
        qa.encode_query = _no_dense

    per_q: List[Dict[str, Any]] = []
    try:
        for g in goldens:
            gt = set(g.get("gt_chunk_ids") or [])
            if not gt:
                # Malformed gold record (no ground-truth chunks) — un-scoreable.
                continue
            raw = qa.hybrid_search(g["question"], collection, retrieval_pool=pool)
            ids = ranked_chunk_ids(raw)
            rank: Optional[int] = next((i + 1 for i, cid in enumerate(ids) if cid in gt), None)
            per_q.append({
                "gold_id": g.get("gold_id"),
                "rank": rank,
                "hits": {k: any(cid in gt for cid in ids[:k]) for k in ks},
                "n_gt": len(gt),
            })
    finally:
        qa.encode_query = orig_encode
    return per_q


def aggregate(per_q: List[Dict[str, Any]], ks: List[int]) -> Dict[str, Any]:
    n = len(per_q) or 1
    hit_at = {k: sum(1 for q in per_q if q["hits"][k]) / n for k in ks}
    mrr = sum((1.0 / q["rank"]) if q["rank"] else 0.0 for q in per_q) / n
    return {"n": len(per_q), "hit_at": hit_at, "mrr": mrr}


def fmt(agg: Dict[str, Any], ks: List[int]) -> str:
    parts = [f"MRR={agg['mrr']:.3f}"] + [f"Hit@{k}={agg['hit_at'][k]*100:5.1f}%" for k in ks]
    return "  ".join(parts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goldens", type=Path, default=DEFAULT_GOLDENS)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--json-out", type=Path, default=None, help="Write full per-query results here.")
    args = p.parse_args(argv)

    import qaEngine as qa  # imported AFTER env is set by the caller

    goldens = load_goldens(args.goldens)
    ks = sorted(args.ks)

    print(f"Backend: EMBED_PROVIDER={qa.EMBED_PROVIDER}  store={qa.PERSIST_DIR}  "
          f"collection={qa.COLLECTION_NAME}")
    scoreable = sum(1 for g in goldens if g.get("gt_chunk_ids"))
    skipped = len(goldens) - scoreable
    print(f"Gold queries: {len(goldens)} total, {scoreable} scoreable"
          + (f", {skipped} skipped (no ground-truth chunks)" if skipped else "")
          + "  (small set - smoke signal, not a verdict)\n")

    hybrid = eval_mode(qa, goldens, ks, bm25_only=False)
    bm25 = eval_mode(qa, goldens, ks, bm25_only=True)

    agg_h = aggregate(hybrid, ks)
    agg_b = aggregate(bm25, ks)

    print(f"  hybrid (embeddings+BM25): {fmt(agg_h, ks)}")
    print(f"  BM25-only baseline:       {fmt(agg_b, ks)}")
    delta = agg_h["mrr"] - agg_b["mrr"]
    print(f"\n  embedding lift (MRR): {delta:+.3f}  "
          f"({'embeddings help' if delta > 0.01 else 'no clear embedding benefit' if abs(delta) <= 0.01 else 'embeddings HURT'})")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"hybrid": {"per_q": hybrid, "agg": agg_h},
             "bm25_only": {"per_q": bm25, "agg": agg_b}}, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
