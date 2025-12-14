#!/usr/bin/env python3
"""
plot_eval.py — Visualize and track retrieval evaluation results (v2)

Features:
  ✓ Reads eval_results_topicaware.json from eval_gold.py
  ✓ Computes per-topic and overall Hit@K, MRR
  ✓ Saves bar chart → eval_plot.png
  ✓ Logs run summary to eval_history.csv (timestamp, averages, model info)
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import csv

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "eval_results_topicaware.json"
HISTORY_PATH = PROJECT_ROOT / "Evaluation" / "eval_history.csv"
TOP_KS = [1, 3, 5]
MODEL_TAG = "BAAI/bge-base-en-v1.5"  # optionally auto-fill from eval_gold later

# -------------------------------------------------------------------
# Load + summarize
# -------------------------------------------------------------------
def load_results(path):
    if not path.exists():
        raise SystemExit(f"❌ Results not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} evaluation entries")
    return data


def summarize(data):
    topic_hits = defaultdict(lambda: {k: 0 for k in TOP_KS})
    topic_totals = defaultdict(int)
    rr_scores = defaultdict(list)

    for item in data:
        topic = item.get("topic", "Unknown")
        rr_scores[topic].append(item.get("reciprocal_rank", 0))
        srcs = item.get("returned_sources", [])
        expected = item.get("expected_source", "")
        aliases = item.get("aliases", [])

        for k in TOP_KS:
            topk = srcs[:k]
            if any(expected in s or any(a in s for a in aliases) for s in topk):
                topic_hits[topic][k] += 1
        topic_totals[topic] += 1

    metrics = {}
    for topic in sorted(topic_totals.keys()):
        totals = topic_totals[topic]
        m = {f"Hit@{k}": topic_hits[topic][k] / totals for k in TOP_KS}
        m["MRR"] = np.mean(rr_scores[topic]) if rr_scores[topic] else 0
        metrics[topic] = m
    return metrics


# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------
def plot_topic_bars(metrics):
    topics = list(metrics.keys())
    x = np.arange(len(topics))
    width = 0.22

    bars1 = [metrics[t]["Hit@1"] for t in topics]
    bars3 = [metrics[t]["Hit@3"] for t in topics]
    bars5 = [metrics[t]["Hit@5"] for t in topics]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, bars1, width, label='Hit@1', alpha=0.8)
    ax.bar(x, bars3, width, label='Hit@3', alpha=0.8)
    ax.bar(x + width, bars5, width, label='Hit@5', alpha=0.8)

    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Topic')
    ax.set_title('Retrieval Performance by Topic')
    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save plot to /Evaluation/eval_plot.png
    eval_dir = PROJECT_ROOT / "Evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "eval_plot.png"
    plt.savefig(out_path, dpi=300)
    print(f"💾 Chart saved to {out_path}")

    plt.show()


# -------------------------------------------------------------------
# Log to CSV
# -------------------------------------------------------------------
def log_to_history(metrics):
    avg = {
        k: np.mean([m[k] for m in metrics.values()])
        for k in [f"Hit@{i}" for i in TOP_KS]
    }
    avg["MRR"] = np.mean([m["MRR"] for m in metrics.values()])

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": timestamp,
        "model": MODEL_TAG,
        "Hit@1": round(avg["Hit@1"], 3),
        "Hit@3": round(avg["Hit@3"], 3),
        "Hit@5": round(avg["Hit@5"], 3),
        "MRR": round(avg["MRR"], 3),
    }

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = HISTORY_PATH.exists()
    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"📘 Logged run to history: {HISTORY_PATH}")

    return avg


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    data = load_results(RESULTS_PATH)
    metrics = summarize(data)

    # Print per-topic table
    print("\n📊 Per-topic metrics:")
    for topic, m in metrics.items():
        print(f"  {topic:<35}  Hit@1: {m['Hit@1']:.2%} | Hit@3: {m['Hit@3']:.2%} | Hit@5: {m['Hit@5']:.2%} | MRR: {m['MRR']:.3f}")

    # Log averages to CSV
    avg = log_to_history(metrics)

    print("\n🌍 Overall averages:")
    for k, v in avg.items():
        print(f"  {k:<6}: {v:.2%}")

    plot_topic_bars(metrics)


if __name__ == "__main__":
    main()
