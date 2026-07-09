#!/usr/bin/env python3
"""
Summarize checked-in retrieval ablation outputs.

Reads legacy eval result JSON files and prints a compact comparison table so
README/docs claims can be regenerated from repository artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [
    PROJECT_ROOT / "eval_results_topicaware.json",
    PROJECT_ROOT / "eval_results_topicaware_reranked.json",
]


def load_rows(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def ndcg_at_k(item: Dict[str, Any], k: int) -> float:
    ndcg = item.get("NDCG", {}) or {}
    return float(ndcg.get(str(k), ndcg.get(k, 0.0)) or 0.0)


def hit_at_k_chunk(item: Dict[str, Any], k: int) -> float:
    gt = {str(x) for x in (item.get("gt_chunk_ids") or [])}
    ranked = [str(x) for x in (item.get("returned_ids") or [])[:k]]
    return 1.0 if any(cid in gt for cid in ranked) else 0.0


def rr_to_hit(rr: float, k: int) -> float:
    return 1.0 if rr and rr >= (1.0 / float(k)) else 0.0


def avg(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def summarize(path: Path) -> Dict[str, Any]:
    rows = load_rows(path)
    return {
        "file": path.name,
        "cases": len(rows),
        "hit_a1": avg([rr_to_hit(float(item.get("RR_article", 0.0) or 0.0), 1) for item in rows]),
        "hit_a3": avg([rr_to_hit(float(item.get("RR_article", 0.0) or 0.0), 3) for item in rows]),
        "hit_a5": avg([rr_to_hit(float(item.get("RR_article", 0.0) or 0.0), 5) for item in rows]),
        "hit_c1": avg([hit_at_k_chunk(item, 1) for item in rows]),
        "hit_c3": avg([hit_at_k_chunk(item, 3) for item in rows]),
        "hit_c5": avg([hit_at_k_chunk(item, 5) for item in rows]),
        "mrr_article": avg([float(item.get("RR_article", 0.0) or 0.0) for item in rows]),
        "mrr_chunk": avg([float(item.get("RR_chunk", 0.0) or 0.0) for item in rows]),
        "ndcg_5": avg([ndcg_at_k(item, 5) for item in rows]),
    }


def format_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def print_markdown(summaries: List[Dict[str, Any]]) -> None:
    print("| Eval file | Cases | Hit@1 article | Hit@3 article | Hit@5 article | Hit@1 chunk | Hit@3 chunk | Hit@5 chunk | MRR article | MRR chunk | NDCG@5 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for item in summaries:
        print(
            "| {file} | {cases} | {hit_a1} | {hit_a3} | {hit_a5} | {hit_c1} | {hit_c3} | {hit_c5} | {mrr_article:.3f} | {mrr_chunk:.3f} | {ndcg_5:.3f} |".format(
                file=item["file"],
                cases=item["cases"],
                hit_a1=format_pct(item["hit_a1"]),
                hit_a3=format_pct(item["hit_a3"]),
                hit_a5=format_pct(item["hit_a5"]),
                hit_c1=format_pct(item["hit_c1"]),
                hit_c3=format_pct(item["hit_c3"]),
                hit_c5=format_pct(item["hit_c5"]),
                mrr_article=item["mrr_article"],
                mrr_chunk=item["mrr_chunk"],
                ndcg_5=item["ndcg_5"],
            )
        )


def print_text(summaries: List[Dict[str, Any]]) -> None:
    for item in summaries:
        print(item["file"])
        print(f"  cases: {item['cases']}")
        print(f"  hit@1 article: {format_pct(item['hit_a1'])}")
        print(f"  hit@3 article: {format_pct(item['hit_a3'])}")
        print(f"  hit@5 article: {format_pct(item['hit_a5'])}")
        print(f"  hit@1 chunk:   {format_pct(item['hit_c1'])}")
        print(f"  hit@3 chunk:   {format_pct(item['hit_c3'])}")
        print(f"  hit@5 chunk:   {format_pct(item['hit_c5'])}")
        print(f"  mrr article:   {item['mrr_article']:.3f}")
        print(f"  mrr chunk:     {item['mrr_chunk']:.3f}")
        print(f"  ndcg@5:        {item['ndcg_5']:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize checked-in eval result files.")
    parser.add_argument("files", nargs="*", default=[str(p) for p in DEFAULT_FILES])
    parser.add_argument("--format", choices=["text", "markdown"], default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(raw) for raw in args.files]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing eval result files: {', '.join(missing)}")
    summaries = [summarize(path) for path in paths]
    if args.format == "markdown":
        print_markdown(summaries)
    else:
        print_text(summaries)


if __name__ == "__main__":
    main()
