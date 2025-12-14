#!/usr/bin/env python3
"""
eval_gold_reranked.py — Evaluate retrieval + reranker vs gold set (v2.0, matches qaEngine v7.6)

Implements the EXACT qaEngine pipeline:

    1. retrieve RETRIEVAL_POOL
    2. apply_bias
    3. group_by_source
    4. trim each group → PER_SOURCE_POOL
    5. rerank within each source (CrossEncoder)
    6. sort within each source
    7. flatten in source-stable order
    8. trim globally → FINAL_LIMIT

Metrics:
    • Hit@K (article, chunk, topic)
    • Precision@K
    • Recall@K
    • MAP@K
    • NDCG@K
    • MRR_article / MRR_chunk
    • Per-topic breakdown
    • JSON + CSV outputs
"""

import json
from pathlib import Path
from statistics import mean
from collections import defaultdict
import csv
import datetime
import sys
import numpy as np

import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config (MATCH qaEngine v7.6)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VDB = PROJECT_ROOT / "VectorDB"
PERSIST_DIR  = str(PROJECT_ROOT / "chroma_store")
COLLECTION_NAME = "msk_chunks"

TOP_KS = [1, 3, 5]

# retrieval knobs identical to qaEngine v7.6
RETRIEVAL_POOL = 100
PER_SOURCE_POOL = 20
FINAL_LIMIT = 50

EMBED_MODEL    = "mixedbread-ai/mxbai-embed-large-v1"
RERANKER_MODEL = "mixedbread-ai/mxbai-reranker-large-v1"
RERANKER_TOP_N = 10

print(f"📦 Embedding model: {EMBED_MODEL}")
print(f"🔁 Reranker model:  {RERANKER_MODEL}")

GOLD_CANDIDATES = [
    PROJECT_ROOT / "Eval" / "gold_set_merged_for_eval.jsonl"
]

RESULTS_JSON = PROJECT_ROOT / "eval_results_topicaware_reranked.json"
HISTORY_CSV  = PROJECT_ROOT / "Evaluation" / "eval_history_reranked.csv"

# ---------------------------------------------------------------------------
# Import qaEngine helpers
# ---------------------------------------------------------------------------

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if str(VDB) not in sys.path:
    sys.path.append(str(VDB))

from qaEngine import apply_bias, group_by_source, maybe_rerank, _backend, OPENAI_MODEL  # noqa


# ---------------------------------------------------------------------------
# Gold loader
# ---------------------------------------------------------------------------

def load_gold():
    for path in GOLD_CANDIDATES:
        if path.exists():
            print(f"📘 Loading gold set: {path}")
            data = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            print(f"✅ Loaded {len(data)} questions")
            return data, path.name
    raise SystemExit("❌ No gold set found.")


# ---------------------------------------------------------------------------
# Basic embed util
# ---------------------------------------------------------------------------

def embed_query(model, text):
    return model.encode([text], normalize_embeddings=True)


# ---------------------------------------------------------------------------
# Metric helpers (unchanged)
# ---------------------------------------------------------------------------

def reciprocal_rank_article(expected_src, aliases, metadatas):
    seen = set()
    rank = 0
    for m in metadatas:
        src = (m.get("source_relpath") or "").strip()
        if src in seen:
            continue
        seen.add(src)
        rank += 1
        if expected_src in src or any(a in src for a in (aliases or [])):
            return 1.0 / rank
    return 0.0


def reciprocal_rank_chunk(gt_ids, ids):
    if not gt_ids:
        return 0.0
    for i, cid in enumerate(ids, start=1):
        if cid in gt_ids:
            return 1.0 / i
    return 0.0


def hit_at_k_article(expected_src, aliases, metas, k):
    seen = set()
    ordered = []
    for m in metas:
        src = (m.get("source_relpath") or "").strip()
        if src not in seen:
            seen.add(src)
            ordered.append(src)
            if len(ordered) >= k:
                break
    return any(expected_src in s or any(a in s for a in aliases or []) for s in ordered)


def hit_at_k_chunk(gt_ids, retrieved_ids, k):
    if not gt_ids:
        return False
    return any(cid in gt_ids for cid in retrieved_ids[:k])


def hit_at_k_topic(topic, metas, k):
    topic_low = (topic or "").lower()
    if not topic_low or topic_low == "general msk/neuro":
        return False
    seen = set()
    ordered = []
    for m in metas:
        src = (m.get("source_relpath") or "").strip()
        if src not in seen:
            seen.add(src)
            ordered.append(src)
            if len(ordered) >= k:
                break
    return any(topic_low in (s or "").lower() for s in ordered)


def precision_at_k(gt, ids, k):
    if not gt or k == 0:
        return 0.0
    hits = sum(1 for cid in ids[:k] if cid in gt)
    return hits / k


def recall_at_k(gt, ids, k):
    if not gt:
        return 0.0
    hits = sum(1 for cid in ids[:k] if cid in gt)
    return hits / len(gt)


def average_precision(gt, ids, k):
    if not gt:
        return 0.0
    hits = 0
    ap = 0.0
    for i, cid in enumerate(ids[:k], start=1):
        if cid in gt:
            hits += 1
            ap += hits / i
    if hits == 0:
        return 0.0
    return ap / min(len(gt), k)


def ndcg_at_k(gt, ids, k):
    def dcg(vals):
        return sum(val / np.log2(i + 2) for i, val in enumerate(vals))
    rels = [1 if cid in gt else 0 for cid in ids[:k]]
    dcg_val = dcg(rels)
    ideal = sorted(rels, reverse=True)
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg_val / ideal_dcg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("🚀 Evaluating reranked retrieval (qaEngine v7.6 pipeline)")

    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    coll = client.get_collection(COLLECTION_NAME)

    gold, gold_name = load_gold()

    hits_article = {k: 0 for k in TOP_KS}
    hits_chunk   = {k: 0 for k in TOP_KS}
    hits_topic   = {k: 0 for k in TOP_KS}
    rr_article   = []
    rr_chunk     = []

    prec_at  = defaultdict(list)
    rec_at   = defaultdict(list)
    map_at   = defaultdict(list)
    ndcg_at  = defaultdict(list)

    topic_stats = defaultdict(lambda: {
        "tot": 0,
        "article": {k: 0 for k in TOP_KS},
        "chunk":   {k: 0 for k in TOP_KS},
        "topic":   {k: 0 for k in TOP_KS},
        "rr_article": [],
        "rr_chunk": [],
        "precision": {k: [] for k in TOP_KS},
        "recall":    {k: [] for k in TOP_KS},
        "map":       {k: [] for k in TOP_KS},
        "ndcg":      {k: [] for k in TOP_KS},
    })

    details = []

    for idx, item in enumerate(gold, 1):
        q        = item["question"]
        topic    = item.get("topic", "Unknown")
        expected = item.get("source_relpath", "")
        aliases  = item.get("aliases", []) or []
        gt_ids   = item.get("gt_chunk_ids") or []

        # ------------------------------------------------------------------
        # 1) baseline retrieval (large pool)
        # ------------------------------------------------------------------
        q_emb = embed_query(model, q)
        baseline_raw = coll.query(query_embeddings=q_emb, n_results=RETRIEVAL_POOL)

        baseline_metas = baseline_raw.get("metadatas", [[]])[0] or []
        baseline_ids   = baseline_raw.get("ids", [[]])[0] or []

        # ------------------------------------------------------------------
        # 2) bias
        # ------------------------------------------------------------------
        biased = apply_bias(q, baseline_raw)

        # convert to clean candidate dicts
        candidates = []
        for it in biased:
            meta = it["meta"] or {}
            cid  = str(meta.get("chunk_id", ""))  # always force string
            candidates.append({
                "meta": meta,
                "dist": float(it["dist"]),
                "chunk_id": cid,
                "text": it["text"],
            })

        # ------------------------------------------------------------------
        # 3) group by source
        # ------------------------------------------------------------------
        grouped = group_by_source(candidates)

        # ------------------------------------------------------------------
        # 4) trim per-group → PER_SOURCE_POOL
        # ------------------------------------------------------------------
        for src in list(grouped.keys()):
            group_sorted = sorted(grouped[src], key=lambda x: x["dist"])
            grouped[src] = group_sorted[:PER_SOURCE_POOL]

        # ------------------------------------------------------------------
        # 5) rerank within each source
        # ------------------------------------------------------------------
        for src in list(grouped.keys()):
            grouped[src] = maybe_rerank(
                question=q,
                candidates=grouped[src],
                backend=_backend,
                openai_model=OPENAI_MODEL,   # <--- changed kwarg name
                top_n=RERANKER_TOP_N,
            )

        # ------------------------------------------------------------------
        # 6) sort within each source, preserve source order
        # ------------------------------------------------------------------
        reranked_items = []
        for src, group in grouped.items():
            reranked_items.extend(sorted(group, key=lambda x: x["dist"]))

        # ------------------------------------------------------------------
        # 7) global FINAL_LIMIT trim
        # ------------------------------------------------------------------
        reranked_items = reranked_items[:FINAL_LIMIT]

        # extract final lists (robust to different reranker output schemas)
        metas = [it.get("meta", {}) for it in reranked_items]

        ids = []
        for i, it in enumerate(reranked_items):
            # common keys used across different components
            if "chunk_id" in it and it["chunk_id"] is not None:
                ids.append(str(it["chunk_id"]))
            elif "id" in it and it["id"] is not None:
                ids.append(str(it["id"]))
            else:
                meta = it.get("meta", {}) or {}
                cid = meta.get("chunk_id") or meta.get("id")
                if cid:
                    ids.append(str(cid))
                else:
                    # final fallback: stable synthetic id (preserves order)
                    ids.append(f"unknown-{idx}-{i}")

        # ------------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------------
        ra = reciprocal_rank_article(expected, aliases, metas)
        rc = reciprocal_rank_chunk(gt_ids, ids)

        rr_article.append(ra)
        if gt_ids:
            rr_chunk.append(rc)

        ts = topic_stats[topic]
        ts["tot"] += 1
        ts["rr_article"].append(ra)
        if gt_ids:
            ts["rr_chunk"].append(rc)

        precisions = {}
        recalls    = {}
        maps       = {}
        ndcgs      = {}

        for k in TOP_KS:
            p = precision_at_k(gt_ids, ids, k)
            r = recall_at_k(gt_ids, ids, k)
            m = average_precision(gt_ids, ids, k)
            n = ndcg_at_k(gt_ids, ids, k)

            precisions[k] = p
            recalls[k]    = r
            maps[k]       = m
            ndcgs[k]      = n

            prec_at[k].append(p)
            rec_at[k].append(r)
            map_at[k].append(m)
            ndcg_at[k].append(n)

            ts["precision"][k].append(p)
            ts["recall"][k].append(r)
            ts["map"][k].append(m)
            ts["ndcg"][k].append(n)

            if hit_at_k_article(expected, aliases, metas, k):
                hits_article[k] += 1
                ts["article"][k] += 1
            if hit_at_k_topic(topic, metas, k):
                hits_topic[k] += 1
                ts["topic"][k] += 1
            if hit_at_k_chunk(gt_ids, ids, k):
                hits_chunk[k] += 1
                ts["chunk"][k] += 1

        # ------------------------------------------------------------------
        # Save detail row
        # ------------------------------------------------------------------
        details.append({
            "query": q,
            "topic": topic,
            "expected_source": expected,
            "aliases": aliases,
            "gt_chunk_ids": gt_ids,

            "returned_ids_base": baseline_ids,
            "returned_sources_base": [
                m.get("source_relpath", "") for m in baseline_metas
            ],

            "returned_ids": ids,
            "returned_sources": [
                m.get("source_relpath", "") for m in metas
            ],

            "RR_article": ra,
            "RR_chunk": rc,
            "precision": precisions,
            "recall": recalls,
            "MAP": maps,
            "NDCG": ndcgs,
            "gold_file": gold_name,
        })

        if idx % 10 == 0 or idx == len(gold):
            print(f"Progress: {idx}/{len(gold)}")

    total = len(gold)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n📊 Hit@K and MRR")
    for k in TOP_KS:
        print(
            f"Hit@{k} Article={hits_article[k]/total:.2%}  "
            f"Chunk={hits_chunk[k]/total:.2%}  "
            f"Topic={hits_topic[k]/total:.2%}"
        )

    print(f"MRR_article = {mean(rr_article):.3f}")
    if rr_chunk:
        print(f"MRR_chunk   = {mean(rr_chunk):.3f}")
    else:
        print("MRR_chunk   = n/a")

    print("\n📐 Precision / Recall / MAP / NDCG")
    for k in TOP_KS:
        print(
            f"@{k}: Precision={mean(prec_at[k]):.2%}  "
            f"Recall={mean(rec_at[k]):.2%}  "
            f"MAP={mean(map_at[k]):.3f}  "
            f"NDCG={mean(ndcg_at[k]):.3f}"
        )

    print("\n🧩 Per-topic breakdown:")
    for topic, st in sorted(topic_stats.items(), key=lambda x: -x[1]["tot"]):
        tot = st["tot"]
        print(f"\nTopic = {topic} (N={tot})")
        for k in TOP_KS:
            print(
                f"  @{k}: HitA={st['article'][k]/tot:.2%}  "
                f"HitC={st['chunk'][k]/tot:.2%}  "
                f"HitT={st['topic'][k]/tot:.2%}  "
                f"Prec={mean(st['precision'][k]):.2%}  "
                f"Rec={mean(st['recall'][k]):.2%}  "
                f"MAP={mean(st['map'][k]):.3f}  "
                f"NDCG={mean(st['ndcg'][k]):.3f}"
            )

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    with RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved reranked details → {RESULTS_JSON}")

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = HISTORY_CSV.exists()

    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gold_file": gold_name,
        "model": EMBED_MODEL,
        "reranker": RERANKER_MODEL,
    }

    for k in TOP_KS:
        row[f"HitA@{k}"] = round(hits_article[k]/total, 4)
        row[f"HitC@{k}"] = round(hits_chunk[k]/total, 4)
        row[f"HitT@{k}"] = round(hits_topic[k]/total, 4)
        row[f"Precision@{k}"] = round(mean(prec_at[k]), 4)
        row[f"Recall@{k}"]    = round(mean(rec_at[k]), 4)
        row[f"MAP@{k}"]       = round(mean(map_at[k]), 4)
        row[f"NDCG@{k}"]      = round(mean(ndcg_at[k]), 4)

    row["MRR_article"] = round(mean(rr_article), 4)
    row["MRR_chunk"]   = round(mean(rr_chunk), 4) if rr_chunk else None

    with HISTORY_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            w.writeheader()
        w.writerow(row)

    print(f"📈 Saved reranked summary → {HISTORY_CSV}")


if __name__ == "__main__":
    main()
