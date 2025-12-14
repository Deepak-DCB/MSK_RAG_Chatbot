#!/usr/bin/env python3
"""
eval_gold.py — Evaluate Chroma retrieval vs gold set (v6, aligned with qaEngine v7.6)

Now correctly mirrors the baseline retrieval path of qaEngine v7.6:

    retrieval_pool → apply_bias → group_by_source
    → per_source_pool → flatten → final_limit
    → (NO reranker) → baseline result list

Everything else (metrics, JSON saving, CSV history) remains unchanged.
"""

import json
from pathlib import Path
from statistics import mean
from collections import defaultdict
import csv
import datetime
import re

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


# ──────────────────────────────────────────────────────────────────────────────
# Config (MATCH qaEngine v7.6)
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
PERSIST_DIR     = str(PROJECT_ROOT / "chroma_store")
COLLECTION_NAME = "msk_chunks"

TOP_KS = [1, 3, 5]

# Retrieval expansion knobs — SAME AS qaEngine v7.6 baseline rules
RETRIEVAL_POOL = 100
PER_SOURCE_POOL = 20
FINAL_LIMIT = 50

GOLD_CANDIDATES = [
    PROJECT_ROOT / "Eval" / "gold_set_merged_for_eval.jsonl",
]

EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
print(f"📦 Using embedding model: {EMBED_MODEL}")

RESULTS_JSON = PROJECT_ROOT / "eval_results_topicaware.json"
HISTORY_CSV  = PROJECT_ROOT / "Evaluation" / "eval_history.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Minimal bias function — IDENTICAL to qaEngine (topic & muscle only)
# (We intentionally exclude section/story heuristics for baseline fairness)
# ──────────────────────────────────────────────────────────────────────────────

TOPIC_PATTERNS = [
    ("thoracic outlet", "thoracic-outlet"),
    ("tos", "thoracic-outlet"),
    ("tmj", "temporomandibular"),
    ("tmd", "temporomandibular"),
    ("pots", "pots"),
    ("postural orthostatic", "pots"),
    ("atlas", "atlas"),
    ("atlantoaxial", "atlanto"),
    ("cci", "atlanto"),
    ("aai", "atlanto"),
    ("jugular", "jugular"),
    ("jos", "jugular"),
    ("lumbar plexus", "lumbar-plexus"),
    ("lpcs", "lumbar-plexus"),
    ("migraine", "migraine"),
]

MUSCLE_TOKENS = [
    "scalene", "scalenes", "trapezius", "levator", "pectoralis",
    "suboccipital", "longus", "sternocleidomastoid", "scm",
    "strength", "strengthen", "stretch", "posture", "kyphosis",
    "hinge", "dyskinesis", "breathing", "mechanics"
]

TOPIC_BONUS  = 0.30
MUSCLE_BONUS = 0.15


def apply_bias(question: str, raw):
    """Simplified qaEngine-style bias (no section/story heuristics)."""

    qlow = question.lower()
    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    dists = raw["distances"][0]

    out = []
    for d, m, base_dist in zip(docs, metas, dists):
        score = float(base_dist)
        txt = d.lower()
        src = (m.get("source_relpath") or "").lower()

        # topic matching
        for needle, hint in TOPIC_PATTERNS:
            if needle in qlow and hint in src:
                score -= TOPIC_BONUS
                break

        # muscle tokens
        if any(tok in txt for tok in MUSCLE_TOKENS):
            score -= MUSCLE_BONUS

        out.append({"text": d, "meta": m, "dist": score})

    out.sort(key=lambda x: x["dist"])
    return out


def group_by_source(items):
    grouped = {}
    for it in items:
        src = it["meta"].get("source_relpath") or ""
        grouped.setdefault(src, []).append(it)
    return grouped


# ──────────────────────────────────────────────────────────────────────────────
# Gold loader
# ──────────────────────────────────────────────────────────────────────────────

def load_gold():
    for path in GOLD_CANDIDATES:
        if path.exists():
            print(f"📘 Loading gold set: {path}")
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except:
                            pass
            print(f"✅ Loaded {len(data)} gold items")
            return data, path.name

    raise SystemExit("❌ No gold set found.")


# ──────────────────────────────────────────────────────────────────────────────
# Embed util
# ──────────────────────────────────────────────────────────────────────────────

def embed_query(model, text: str) -> np.ndarray:
    return model.encode([text], normalize_embeddings=True)


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers (UNCHANGED)
# ──────────────────────────────────────────────────────────────────────────────

def reciprocal_rank_article(expected_src: str, aliases: list, metadatas: list) -> float:
    seen = set()
    rank = 0
    for m in metadatas:
        src = (m.get("source_relpath") or "").strip()
        if src in seen:
            continue
        seen.add(src)
        rank += 1
        if expected_src in src or any(a in src for a in aliases or []):
            return 1.0 / rank
    return 0.0


def reciprocal_rank_chunk(gt_chunk_ids: list, ids: list) -> float:
    if not gt_chunk_ids:
        return 0.0
    for i, cid in enumerate(ids, start=1):
        if cid in gt_chunk_ids:
            return 1.0 / i
    return 0.0


def hit_at_k_article(expected_src, aliases, metas, k):
    seen = set()
    topk = []
    for m in metas:
        src = (m.get("source_relpath") or "").strip()
        if src not in seen:
            seen.add(src)
            topk.append(src)
            if len(topk) >= k:
                break
    return any(expected_src in s or any(a in s for a in aliases or []) for s in topk)


def hit_at_k_chunk(gt_chunk_ids, ids, k):
    if not gt_chunk_ids:
        return False
    return any(cid in gt_chunk_ids for cid in ids[:k])


def hit_at_k_topic(topic, metas, k):
    topic_low = (topic or "").lower()
    if not topic_low or topic_low == "general msk/neuro":
        return False

    seen = set()
    topk = []
    for m in metas:
        src = (m.get("source_relpath") or "").strip()
        if src not in seen:
            seen.add(src)
            topk.append(src)
            if len(topk) >= k:
                break
    return any(topic_low in (s or "").lower() for s in topk)


def precision_at_k(gt_chunk_ids, retrieved_ids, k):
    if k == 0:
        return 0.0
    if not gt_chunk_ids:
        return 0.0
    topk = retrieved_ids[:k]
    hits = sum(1 for cid in topk if cid in gt_chunk_ids)
    return hits / k


def recall_at_k(gt_chunk_ids, retrieved_ids, k):
    if not gt_chunk_ids:
        return 0.0
    topk = retrieved_ids[:k]
    hits = sum(1 for cid in topk if cid in gt_chunk_ids)
    return hits / len(gt_chunk_ids)


def average_precision(gt_chunk_ids, retrieved_ids, k):
    if not gt_chunk_ids:
        return 0.0

    topk = retrieved_ids[:k]
    ap = 0.0
    hits = 0

    for i, cid in enumerate(topk, start=1):
        if cid in gt_chunk_ids:
            hits += 1
            ap += hits / i

    if hits == 0:
        return 0.0

    return ap / min(len(gt_chunk_ids), k)


def ndcg_at_k(gt_chunk_ids, retrieved_ids, k):
    def dcg(vals):
        return sum(val / np.log2(idx + 2) for idx, val in enumerate(vals))

    topk = retrieved_ids[:k]
    rels = [1 if cid in gt_chunk_ids else 0 for cid in topk]
    dcg_val = dcg(rels)

    ideal = sorted(rels, reverse=True)
    ideal_dcg = dcg(ideal)

    if ideal_dcg == 0:
        return 0.0

    return dcg_val / ideal_dcg


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation (UPDATED BASELINE SECTION)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("🚀 Evaluating ChromaDB retrieval quality")
    print(f"🔧 Chroma dir: {PERSIST_DIR}")
    print(f"🔧 Retrieval pool={RETRIEVAL_POOL}, per_source_pool={PER_SOURCE_POOL}, final_limit={FINAL_LIMIT}\n")

    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    coll = client.get_collection(COLLECTION_NAME)

    gold, gold_name = load_gold()

    hits_article = {k: 0 for k in TOP_KS}
    hits_chunk   = {k: 0 for k in TOP_KS}
    hits_topic   = {k: 0 for k in TOP_KS}
    rr_article   = []
    rr_chunk     = []

    prec_at = defaultdict(list)
    rec_at  = defaultdict(list)
    map_at  = defaultdict(list)
    ndcg_at = defaultdict(list)

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

    for i, item in enumerate(gold, 1):
        q        = item["question"]
        topic    = item.get("topic", "Unknown")
        expected = item.get("source_relpath", "") or ""
        aliases  = item.get("aliases", []) or []
        gt_chunks = item.get("gt_chunk_ids") or []

        q_emb = embed_query(model, q)

        # ───────────────────────────────────────────────────────
        # UPDATED BASELINE PIPELINE (MATCHES qaEngine v7.6)
        # ───────────────────────────────────────────────────────

        # 1) Large retrieval
        raw = coll.query(query_embeddings=q_emb, n_results=RETRIEVAL_POOL)

        # 2) Apply topic/muscle bias
        biased = apply_bias(q, raw)

        # 3) Group by source
        grouped = group_by_source(biased)

        # 4) Take up to PER_SOURCE_POOL per source
        trimmed = []
        for src, group in grouped.items():
            group_sorted = sorted(group, key=lambda x: x["dist"])
            trimmed.extend(group_sorted[:PER_SOURCE_POOL])

        # 5) Sort globally, limit to FINAL_LIMIT
        trimmed.sort(key=lambda x: x["dist"])
        trimmed = trimmed[:FINAL_LIMIT]

        # 6) Build baseline list (NO reranker)
        baseline_metas = [it["meta"] for it in trimmed]
        baseline_ids   = [it["meta"].get("chunk_id") for it in trimmed]

        # ───────────────────────────────────────────────────────
        # Evaluation metrics (UNCHANGED)
        # ───────────────────────────────────────────────────────

        rr_a = reciprocal_rank_article(expected, aliases, baseline_metas)
        rr_c = reciprocal_rank_chunk(gt_chunks, baseline_ids)

        rr_article.append(rr_a)
        if gt_chunks:
            rr_chunk.append(rr_c)

        topic_stats[topic]["tot"] += 1
        topic_stats[topic]["rr_article"].append(rr_a)
        if gt_chunks:
            topic_stats[topic]["rr_chunk"].append(rr_c)

        precisions = {}
        recalls    = {}
        maps       = {}
        ndcgs      = {}

        for k in TOP_KS:
            p = precision_at_k(gt_chunks, baseline_ids, k)
            r = recall_at_k(gt_chunks, baseline_ids, k)
            a = average_precision(gt_chunks, baseline_ids, k)
            n = ndcg_at_k(gt_chunks, baseline_ids, k)

            precisions[k] = p
            recalls[k]    = r
            maps[k]       = a
            ndcgs[k]      = n

            prec_at[k].append(p)
            rec_at[k].append(r)
            map_at[k].append(a)
            ndcg_at[k].append(n)

            topic_stats[topic]["precision"][k].append(p)
            topic_stats[topic]["recall"][k].append(r)
            topic_stats[topic]["map"][k].append(a)
            topic_stats[topic]["ndcg"][k].append(n)

            if hit_at_k_article(expected, aliases, baseline_metas, k):
                hits_article[k] += 1
                topic_stats[topic]["article"][k] += 1
            if hit_at_k_topic(topic, baseline_metas, k):
                hits_topic[k] += 1
                topic_stats[topic]["topic"][k] += 1
            if gt_chunks and hit_at_k_chunk(gt_chunks, baseline_ids, k):
                hits_chunk[k] += 1
                topic_stats[topic]["chunk"][k] += 1

        # ───────────────────────────────────────────────────────
        # JSON detail entry
        # ───────────────────────────────────────────────────────

        details.append({
            "query": q,
            "topic": topic,
            "expected_source": expected,
            "aliases": aliases,
            "gt_chunk_ids": gt_chunks,

            "returned_ids_base": baseline_ids,
            "returned_sources_base": [
                m.get("source_relpath","") for m in baseline_metas
            ],

            # For compatibility with eval_gold_reranked.json
            "returned_ids": baseline_ids,
            "returned_sources": [
                m.get("source_relpath","") for m in baseline_metas
            ],

            "RR_article": rr_a,
            "RR_chunk": rr_c,
            "precision": precisions,
            "recall": recalls,
            "MAP": maps,
            "NDCG": ndcgs,
            "gold_file": gold_name,
        })

        if i % 10 == 0 or i == len(gold):
            print(f"Progress: {i}/{len(gold)}")

    total = len(gold) or 1

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

    print("\n📐 Precision / Recall / MAP / NDCG (chunk-level)")
    for k in TOP_KS:
        P = mean(prec_at[k]) if prec_at[k] else 0
        R = mean(rec_at[k])  if rec_at[k]  else 0
        A = mean(map_at[k])  if map_at[k]  else 0
        N = mean(ndcg_at[k]) if ndcg_at[k] else 0
        print(f"@{k}: Precision={P:.2%}  Recall={R:.2%}  MAP={A:.3f}  NDCG={N:.3f}")

    print("\n🧩 Per-topic breakdown:")
    for topic, st in sorted(topic_stats.items(), key=lambda x: -x[1]["tot"]):
        tot = st["tot"]
        print(f"\nTopic = {topic} (N={tot})")

        for k in TOP_KS:
            print(
                f"  @{k}: "
                f"HitA={st['article'][k]/tot:.2%}  "
                f"HitC={st['chunk'][k]/tot:.2%}  "
                f"HitT={st['topic'][k]/tot:.2%}  "
                f"Prec={mean(st['precision'][k]):.2%}  "
                f"Rec={mean(st['recall'][k]):.2%}  "
                f"MAP={mean(st['map'][k]):.3f}  "
                f"NDCG={mean(st['ndcg'][k]):.3f}"
            )

    # save JSON
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved detailed results → {RESULTS_JSON}")

    # save CSV
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gold_file": gold_name,
        "model": EMBED_MODEL,
    }

    for k in TOP_KS:
        row[f"HitA@{k}"] = round(hits_article[k]/total, 4)
        row[f"HitC@{k}"] = round(hits_chunk[k]/total,   4)
        row[f"HitT@{k}"] = round(hits_topic[k]/total,   4)

    row["MRR_article"] = round(mean(rr_article), 4)
    row["MRR_chunk"]   = round(mean(rr_chunk),   4) if rr_chunk else None

    for k in TOP_KS:
        row[f"Precision@{k}"] = round(mean(prec_at[k]), 4)
        row[f"Recall@{k}"]    = round(mean(rec_at[k]), 4)
        row[f"MAP@{k}"]       = round(mean(map_at[k]), 4)
        row[f"NDCG@{k}"]      = round(mean(ndcg_at[k]), 4)

    exists = HISTORY_CSV.exists()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"📈 Saved summary to → {HISTORY_CSV}")


if __name__ == "__main__":
    main()
