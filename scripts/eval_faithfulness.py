#!/usr/bin/env python3
"""
eval_faithfulness.py — measure how grounded the pipeline's answers actually are.

Runs the RAG pipeline on gold questions, then scores each answer's faithfulness
(fraction of claims supported by the retrieved evidence) with an LLM judge. This
turns "honesty" into a tracked number.

Runs key-free — local embeddings + Groq generation + Groq judge:

    MSK_EMBED_PROVIDER=local \
    MSK_CHROMA_DIR=<abs path to chroma_store_local> \
    MSK_COLLECTION=msk_chunks \
    python scripts/eval_faithfulness.py --max-cases 10 --key-file groq.key

Notes:
  * Generation falls to Groq automatically because no OpenAI key is present (the
    OpenAI attempt fails -> free-provider chain). evidence_only answers (retrieval
    or generation degraded) are grounded by construction; they are reported but the
    headline number focuses on real LLM-generated answers.
  * The judge is itself an LLM — treat the score as a strong signal, not ground
    truth. Spot-check flagged unsupported claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT / "VectorDB", PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DEFAULT_QUESTIONS = PROJECT_ROOT / "datasets" / "retrieval-goldens-generated.jsonl"

import faithfulness as F  # noqa: E402


def load_questions(path: Path, limit: int) -> List[str]:
    qs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            qs.append(json.loads(line)["question"])
    return qs[:limit] if limit else qs


def evidence_text(contexts: List[Dict[str, Any]]) -> str:
    return "\n\n".join((c.get("text") or "") for c in (contexts or []))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--max-cases", type=int, default=10)
    p.add_argument("--model", default="openai/gpt-oss-120b")
    p.add_argument("--key-file", type=Path, default=None)
    p.add_argument("--sleep", type=float, default=6.0, help="Seconds between cases (TPM throttle).")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="List cases; no API calls.")
    args = p.parse_args(argv)

    questions = load_questions(args.questions, args.max_cases)
    print(f"Faithfulness eval: {len(questions)} questions from {args.questions.name}")
    if args.dry_run:
        for q in questions[:5]:
            print("  would score:", q[:70])
        print("DRY RUN — no API calls, no writes.")
        return 0

    import qaEngine as qa
    from openai import OpenAI

    key = (args.key_file.read_text().strip() if args.key_file
           else os.getenv("GROQ_API_KEY", "").strip())
    if not key:
        print("No Groq key. Set GROQ_API_KEY or pass --key-file.")
        return 2
    os.environ["GROQ_API_KEY"] = key  # so the pipeline's generation falls to Groq
    judge = F.make_groq_judge(
        OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1"), args.model)

    print(f"Backend: EMBED_PROVIDER={qa.EMBED_PROVIDER}  store={qa.PERSIST_DIR}\n")

    results: List[Dict[str, Any]] = []
    for i, q in enumerate(questions):
        try:
            res = qa.agentic_run(q)
        except Exception as exc:
            print(f"  [{i+1}/{len(questions)}] pipeline error: {type(exc).__name__}: {exc}")
            continue
        answer = res.get("answer", "")
        evidence = evidence_text(res.get("contexts"))
        sc = F.score_answer(answer, evidence, judge=judge)
        sc.update({"q": q, "answer_mode": res.get("answer_mode"),
                   "retrieval_confidence": res.get("retrieval_confidence"),
                   "answer": answer, "evidence": evidence})  # saved for offline re-judging/debug
        results.append(sc)
        s = sc["score"]
        print(f"  [{i+1}/{len(questions)}] score={'n/a' if s is None else f'{s:.2f}'}  "
              f"claims={sc['n_claims']}  mode={sc['answer_mode']}")
        time.sleep(args.sleep)

    llm = [r for r in results if r["score"] is not None and str(r["answer_mode"]).startswith("llm:")]
    evidence_only = [r for r in results if str(r["answer_mode"]) == "evidence_only"]
    scored = [r for r in results if r["score"] is not None]

    print("\n=== Faithfulness ===")
    if scored:
        mean = sum(r["score"] for r in scored) / len(scored)
        tc = sum(r["n_claims"] for r in scored)
        tok = sum(r["n_supported"] for r in scored)
        print(f"  mean answer faithfulness: {mean:.3f}  over {len(scored)} scored answers")
        print(f"  claim-level groundedness: {tok}/{tc} = {tok/tc:.3f}" if tc else "  (no claims)")
    if llm:
        lm = sum(r["score"] for r in llm) / len(llm)
        print(f"  LLM-generated only:       {lm:.3f}  over {len(llm)} answers")
    print(f"  answer modes: {len(llm)} llm, {len(evidence_only)} evidence_only, "
          f"{len(results) - len(llm) - len(evidence_only)} other")

    worst = sorted((r for r in scored if r["unsupported"]), key=lambda r: r["score"])[:3]
    if worst:
        print("\n  least-grounded answers:")
        for r in worst:
            print(f"    score={r['score']:.2f}  Q: {r['q'][:60]}")
            for u in r["unsupported"][:2]:
                print(f"       UNSUPPORTED: {u[:100]}")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
