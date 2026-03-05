#!/usr/bin/env python3
"""
run_eval.py — Quick retrieval evaluation using the production pipeline.

Runs gold set queries through qaEngine.run_qa() (without generation)
and measures Hit@K, MRR, and retrieval confidence.

Usage:
    python scripts/run_eval.py
"""

import json
import sys
from pathlib import Path
from statistics import mean

# Add parent dir to path so we can import qaEngine
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "VectorDB"))

from qaEngine import run_qa, QAConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = PROJECT_ROOT / "Eval" / "gold_set_merged_for_eval.jsonl"

TOP_KS = [1, 3, 5]


def load_gold():
    if not GOLD_PATH.exists():
        raise SystemExit(f"❌ Gold set not found: {GOLD_PATH}")

    data = []
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"✅ Loaded {len(data)} gold items from {GOLD_PATH.name}")
    return data


def evaluate():
    gold = load_gold()

    # Disable generation (only test retrieval)
    cfg = QAConfig(generate_answer=False, use_reranker=True)

    hits_article = {k: 0 for k in TOP_KS}
    rr_list = []
    confidences = []

    for i, item in enumerate(gold, 1):
        q = item["question"]
        expected = (item.get("source_relpath") or "").strip()
        aliases = item.get("aliases", []) or []

        result = run_qa(q, config=cfg)

        # Check returned contexts
        contexts = result.get("contexts", [])
        confidence = result.get("retrieval_confidence", 0.0)
        confidences.append(confidence)

        # Build source list from contexts
        sources = []
        seen = set()
        for ctx in contexts:
            src = (ctx.get("meta", {}).get("source_relpath") or "").strip()
            if src and src not in seen:
                seen.add(src)
                sources.append(src)

        # MRR
        rr = 0.0
        for rank, src in enumerate(sources, start=1):
            if expected in src or any(a in src for a in aliases):
                rr = 1.0 / rank
                break
        rr_list.append(rr)

        # Hit@K
        for k in TOP_KS:
            top_k_srcs = sources[:k]
            if any(expected in s or any(a in s for a in aliases) for s in top_k_srcs):
                hits_article[k] += 1

        if i % 5 == 0 or i == len(gold):
            print(f"  Progress: {i}/{len(gold)}")

    total = len(gold) or 1

    print("\n" + "=" * 60)
    print("📊 RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n🔧 Pipeline: hybrid search + BM25 + multi-query + reranker")
    print(f"📏 Gold set: {len(gold)} queries\n")

    print("Hit@K (article-level):")
    for k in TOP_KS:
        rate = hits_article[k] / total
        print(f"  Hit@{k}: {rate:.1%} ({hits_article[k]}/{total})")

    mrr = mean(rr_list) if rr_list else 0
    print(f"\nMRR (article): {mrr:.3f}")

    avg_conf = mean(confidences) if confidences else 0
    print(f"Avg retrieval confidence: {avg_conf:.3f}")

    print("\n" + "=" * 60)
    return {
        "hit_at_k": {k: hits_article[k] / total for k in TOP_KS},
        "mrr": mrr,
        "avg_confidence": avg_conf,
    }


if __name__ == "__main__":
    results = evaluate()
