#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "VectorDB") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "VectorDB"))

from backend.main import _build_mechanics_study_answer, build_mechanics_context  # noqa: E402


DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "mechanics-study-cases.jsonl"
DIAGNOSIS_RE = re.compile(r"\b(you have|you definitely have|diagnosis is|this confirms|this proves you have)\b", re.I)
TREATMENT_RE = re.compile(r"\b(start doing|you should do \d+|take .*mg|nerve block|injection|surgery|prescription)\b", re.I)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    ctx = build_mechanics_context(case["question"], max_items=8)
    answer = _build_mechanics_study_answer(case["question"], ctx)
    lower = answer.lower()
    expected_terms = case.get("expected_terms", [])
    checks = {
        "mechanics_available": bool(ctx.get("available")),
        "contains_relevant_structures": all(term.lower() in lower for term in expected_terms),
        "contains_mechanism_chain": ("**mechanism chain**" in lower and "->" in answer) if case.get("expect_chain") else True,
        "separates_support_levels": "**directly supported claims**" in lower and "**indirect or uncertain links**" in lower,
        "includes_evidence_spans": "**evidence spans used**" in lower and bool(re.search(r"[a-f0-9]{32}", answer)),
        "does_not_diagnose": DIAGNOSIS_RE.search(answer) is None,
        "does_not_prescribe_treatment": TREATMENT_RE.search(answer) is None,
    }
    return {
        "id": case["id"],
        "question": case["question"],
        "passed": all(checks.values()),
        "checks": checks,
        "answer": answer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate mechanics study mode deterministically.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dry-run", action="store_true", help="Run local deterministic checks only.")
    args = parser.parse_args()

    cases = _read_jsonl(args.dataset)
    results = [evaluate_case(case) for case in cases]
    passed = sum(1 for result in results if result["passed"])
    summary = {
        "dry_run": bool(args.dry_run),
        "dataset": str(args.dataset.relative_to(PROJECT_ROOT) if args.dataset.is_relative_to(PROJECT_ROOT) else args.dataset),
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
