#!/usr/bin/env python3
"""
model_comparison.py — Compare multiple Chroma stores (one per embedding model).

Features
--------
• Simple top-of-file MODELS list for quick edits (path, collection, tag)
• Loads gold set (question → gt_chunk_id), queries each store, and computes:
    - Hit@K
    - Recall@K (identical to Hit@K when there is a single relevant item)
    - Precision@K (Hit@K divided by K)
    - MRR
• Saves: eval_history.csv (append), model_comparison_summary.csv, model_comparison.png
• Optional per-question details CSV

Usage
-----
# Use MODELS (defined below)
python model_comparison.py --gold ../gold_set.jsonl

# Override with explicit stores (path:collection:tag)
python model_comparison.py --gold ../gold_set.jsonl \
  --stores "ChromaStore_MXBAI:msk_mxbai:MXBAI,ChromaStore_BGEbase:msk_bgebase:BGE-base"

# Auto-discover subfolders as stores (same collection name for all)
python model_comparison.py --gold ../gold_set.jsonl --auto-root . --collection-name msk_chunks
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import chromadb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================== EDIT HERE: DEFAULT MODELS ===========================

# Each tuple: (path_to_chroma_store_folder, collection_name, short_tag)
# Feel free to add/remove lines; this list is used if you don't pass --stores/--auto-root.
MODELS: List[Tuple[str, str, str]] = [
    ("ChromaStore_MXBAI",     "msk_mxbai",     "MXBAI"),
    ("ChromaStore_BGEbase",   "msk_bgebase",   "BGE-base"),
    # ("ChromaStore_Instructor","msk_instructor","Instructor"),  # example
]

# =================================================================================


DEFAULT_TOPKS = [1, 3, 5]
DEFAULT_OUTDIR = Path("Evaluation")
DEFAULT_PLOT_PATH = "model_comparison.png"
DEFAULT_HISTORY_CSV = "eval_history.csv"
DEFAULT_DETAILS_CSV = "eval_details.csv"


@dataclass
class StoreSpec:
    path: Path
    collection: str
    tag: str


# ----------------------------- Gold loader --------------------------------------

def load_gold(gold_path: Path) -> pd.DataFrame:
    """
    Load gold set (JSONL or CSV). Must contain:
      - question text: 'question' or 'query' (or 'prompt', 'q')
      - ground-truth chunk id: 'gt_chunk_id' or 'chunk_id' (or 'answer_chunk_id', 'target_chunk_id')
    """
    if not gold_path.exists():
        raise SystemExit(f"❌ Gold file not found: {gold_path}")

    if gold_path.suffix.lower() in (".jsonl", ".json"):
        rows = []
        with gold_path.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    print(f"⚠️  Gold JSON decode error at line {i}: {e}")
        df = pd.DataFrame(rows)
    elif gold_path.suffix.lower() == ".csv":
        df = pd.read_csv(gold_path)
    else:
        raise SystemExit(f"❌ Unsupported gold format: {gold_path.suffix}")

    q_col = next((c for c in ["question", "query", "prompt", "q"] if c in df.columns), None)
    gt_col = next((c for c in ["gt_chunk_id", "chunk_id", "answer_chunk_id", "target_chunk_id"] if c in df.columns), None)

    if not q_col or not gt_col:
        raise SystemExit("❌ Gold must include question (question/query) and ground-truth id (gt_chunk_id/chunk_id).")

    df = df[[q_col, gt_col]].rename(columns={q_col: "question", gt_col: "gt_chunk_id"})
    before = len(df)
    df = df.dropna(subset=["question", "gt_chunk_id"]).copy()
    if len(df) == 0:
        raise SystemExit("❌ Gold set is empty after filtering.")
    dropped = before - len(df)
    if dropped:
        print(f"⚠️  Dropped {dropped} gold rows missing question/gt_chunk_id.")

    df["question"] = df["question"].astype(str)
    df["gt_chunk_id"] = df["gt_chunk_id"].astype(str)
    print(f"✅ Gold loaded: {len(df)} questions")
    return df


# ----------------------------- Store list parsing -------------------------------

def parse_store_list(stores_arg: Optional[str]) -> List[StoreSpec]:
    """
    Parse --stores "path:collection:tag,other:collection:tag".
    """
    specs: List[StoreSpec] = []
    if not stores_arg:
        return specs
    for item in [x.strip() for x in stores_arg.split(",") if x.strip()]:
        parts = item.split(":")
        if len(parts) == 1:
            p = Path(parts[0]); coll = "msk_chunks"; tag = p.name
        elif len(parts) == 2:
            p = Path(parts[0]); coll = parts[1]; tag = p.name
        else:
            p = Path(parts[0]); coll = parts[1]; tag = parts[2]
        specs.append(StoreSpec(p, coll, tag))
    return specs


def auto_discover_stores(root: Path, collection_name: str) -> List[StoreSpec]:
    """
    Auto-discover subfolders under root as stores, using a shared collection name.
    """
    specs: List[StoreSpec] = []
    for child in root.iterdir():
        if child.is_dir():
            specs.append(StoreSpec(child, collection_name, child.name))
    if not specs:
        print(f"⚠️  No subdirectories found under {root} to treat as Chroma stores.")
    return specs


def specs_from_default_models() -> List[StoreSpec]:
    return [StoreSpec(Path(p), c, t) for (p, c, t) in MODELS]


# ----------------------------- Evaluation core ----------------------------------

def evaluate_store(
    store: StoreSpec,
    gold_df: pd.DataFrame,
    topks: List[int],
    n_results: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Evaluate a single store against the gold set.
    Returns per-question DataFrame and a metrics dict with Hit@K, Recall@K, Precision@K, MRR, latency.
    """
    n_results = n_results or max(topks)
    client = chromadb.PersistentClient(path=str(store.path))
    coll = client.get_collection(store.collection)

    # accumulators
    hits_at_k = {k: [] for k in topks}        # 1 if gt in top-k else 0
    prec_at_k = {k: [] for k in topks}        # (1/k) if hit else 0
    rr_list: List[float] = []
    latencies: List[float] = []

    per_rows = []

    for _, row in gold_df.iterrows():
        q = row["question"]
        gt = str(row["gt_chunk_id"])

        t0 = perf_counter()
        res = coll.query(query_texts=[q], n_results=n_results)
        t1 = perf_counter()
        lat = (t1 - t0) * 1000.0
        latencies.append(lat)

        ids: List[str] = res.get("ids", [[]])[0]

        # rank and metrics
        rank = None
        for idx, cid in enumerate(ids, start=1):
            if str(cid) == gt:
                rank = idx
                break

        rr = 1.0 / rank if rank is not None else 0.0
        rr_list.append(rr)

        for k in topks:
            hit = 1.0 if (rank is not None and rank <= k) else 0.0
            hits_at_k[k].append(hit)            # Recall@K (for single relevant item)
            prec_at_k[k].append(hit / k)        # Precision@K

        per_rows.append({
            "tag": store.tag,
            "collection": store.collection,
            "store_path": str(store.path),
            "question": q,
            "gt_chunk_id": gt,
            "rank": rank if rank is not None else 0,
            "rr": rr,
            "latency_ms": lat,
            "retrieved_ids": ids,
        })

    # summaries
    metrics: Dict[str, float] = {}
    for k in topks:
        metrics[f"Hit@{k}"] = float(np.mean(hits_at_k[k])) if hits_at_k[k] else 0.0
        metrics[f"Recall@{k}"] = metrics[f"Hit@{k}"]  # single relevant item
        metrics[f"Precision@{k}"] = float(np.mean(prec_at_k[k])) if prec_at_k[k] else 0.0

    metrics["MRR"] = float(np.mean(rr_list)) if rr_list else 0.0
    metrics["latency_mean_ms"] = float(np.mean(latencies)) if latencies else 0.0
    metrics["latency_p95_ms"] = float(np.percentile(latencies, 95)) if latencies else 0.0

    per_q_df = pd.DataFrame(per_rows)
    return per_q_df, metrics


# ----------------------------- Plot & persistence -------------------------------

def append_history(
    outdir: Path,
    tag: str,
    collection: str,
    store_path: Path,
    metrics: Dict[str, float],
    topks: List[int],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / DEFAULT_HISTORY_CSV

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "collection": collection,
        "store_path": str(store_path),
        "MRR": metrics["MRR"],
        "latency_mean_ms": metrics["latency_mean_ms"],
        "latency_p95_ms": metrics["latency_p95_ms"],
        **{f"Hit@{k}": metrics[f"Hit@{k}"] for k in topks},
        **{f"Recall@{k}": metrics[f"Recall@{k}"] for k in topks},
        **{f"Precision@{k}": metrics[f"Precision@{k}"] for k in topks},
    }

    df_row = pd.DataFrame([record])
    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, index=False)
    print(f"📝 Appended results → {csv_path}")


def save_details(outdir: Path, details_df: pd.DataFrame) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / DEFAULT_DETAILS_CSV
    details_df.to_csv(path, index=False)
    print(f"🧾 Per-question details → {path}")


def plot_comparison(
    outdir: Path,
    summary_table: pd.DataFrame,
    topks: List[int],
    filename: str = DEFAULT_PLOT_PATH,
) -> Path:
    """
    Bar plot of Recall@K with an MRR line (Precision@K is in the CSVs).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    tags = summary_table["tag"].tolist()
    mrrs = summary_table["MRR"].tolist()

    # Bar groups for Recall@K
    x = np.arange(len(tags))
    width = 0.13 if len(topks) >= 3 else 0.18
    offsets = [i - (len(topks) - 1) / 2 * width for i in range(len(topks))]

    plt.figure(figsize=(11, 6))
    for i, k in enumerate(topks):
        plt.bar(x + offsets[i], summary_table[f"Recall@{k}"], width=width, label=f"Recall@{k}")

    # MRR line
    plt.plot(x, mrrs, marker="o", linewidth=2, label="MRR")

    plt.xticks(x, tags)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Model Comparison — Recall@K (bars) and MRR (line)")
    plt.legend()
    plt.tight_layout()

    out_path = outdir / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"📈 Plot saved → {out_path}")
    return out_path


# ----------------------------- CLI / Orchestration ------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate multiple Chroma stores against a gold set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gold", type=Path, required=True, help="Path to gold_set.jsonl or .csv")
    p.add_argument(
        "--stores",
        type=str,
        default="",
        help="Comma-separated list: 'path:collection:tag'. Example: "
             "'ChromaStore_MXBAI:msk_mxbai:MXBAI,ChromaStore_BGEbase:msk_bgebase:BGE-base'",
    )
    p.add_argument("--auto-root", type=Path, default=None,
                   help="Auto-discover subfolders as stores.")
    p.add_argument("--collection-name", type=str, default="msk_chunks",
                   help="Collection name for auto-discovered stores.")
    p.add_argument("--topks", type=int, nargs="+", default=DEFAULT_TOPKS,
                   help="K values for Hit/Recall/Precision.")
    p.add_argument("--n-results", type=int, default=None,
                   help="Override n_results (defaults to max(topks)).")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                   help="Directory for CSVs and plots.")
    p.add_argument("--details", action="store_true",
                   help="Save per-question details CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Print and use default MODELS if --stores/--auto-root not provided
    default_specs = specs_from_default_models()
    print("🔧 Models configured at top of file:")
    for s in default_specs:
        print(f"  • {s.tag}: path={s.path} | collection={s.collection}")

    gold_df = load_gold(args.gold)

    # Resolve store specs: CLI overrides defaults if provided
    specs: List[StoreSpec] = []
    if args.stores.strip():
        specs.extend(parse_store_list(args.stores))
    elif args.auto_root:
        specs.extend(auto_discover_stores(args.auto_root, args.collection_name))
    else:
        specs.extend(default_specs)

    # De-dup in case of overlap
    uniq = {}
    for s in specs:
        uniq[(s.path.resolve(), s.collection, s.tag)] = s
    specs = list(uniq.values())

    if not specs:
        raise SystemExit("❌ No stores provided or discovered.")

    print("\n🔎 Evaluating stores:")
    for s in specs:
        print(f"  • {s.tag} → path={s.path} | collection={s.collection}")

    summaries = []
    all_details = []

    for s in specs:
        print(f"\n=== Evaluating: {s.tag} ===")
        per_df, metrics = evaluate_store(s, gold_df, topks=args.topks, n_results=args.n_results)

        row = {"tag": s.tag, "collection": s.collection, "store_path": str(s.path)}
        for k in args.topks:
            row[f"Hit@{k}"] = metrics[f"Hit@{k}"]
            row[f"Recall@{k}"] = metrics[f"Recall@{k}"]
            row[f"Precision@{k}"] = metrics[f"Precision@{k}"]
        row["MRR"] = metrics["MRR"]
        row["latency_mean_ms"] = metrics["latency_mean_ms"]
        row["latency_p95_ms"] = metrics["latency_p95_ms"]
        summaries.append(row)

        append_history(args.outdir, s.tag, s.collection, s.path, metrics, args.topks)

        if args.details:
            all_details.append(per_df)

    summary_df = pd.DataFrame(summaries).sort_values("MRR", ascending=False).reset_index(drop=True)
    print("\n🏁 Summary:\n", summary_df)

    # Save plot (Recall@K bars + MRR line)
    plot_comparison(args.outdir, summary_df, args.topks, filename=DEFAULT_PLOT_PATH)

    # Save summary CSV
    summary_csv = args.outdir / "model_comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"🧮 Summary CSV → {summary_csv}")

    # Save details CSV (optional)
    if args.details and all_details:
        details_df = pd.concat(all_details, ignore_index=True)
        save_details(args.outdir, details_df)


if __name__ == "__main__":
    main()
