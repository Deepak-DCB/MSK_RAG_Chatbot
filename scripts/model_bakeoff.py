#!/usr/bin/env python3
"""
model_bakeoff.py — automated assessment of the best default generation model + settings.

Runs the SAME gold questions through several candidate (provider, model, num_predict)
configs via the REAL pipeline, and scores each on the axes that matter for a
domain-constrained medical RAG:

  * faithfulness — fraction of answer claims supported by the retrieved evidence
                   (higher = better; the core "is it true?" metric)
  * refusal_rate — fraction that declined or degraded to evidence-only
  * llm_rate     — fraction that produced a real model answer (not a degraded fallback);
                   low llm_rate = the model is unreliable (empties / errors / rate limits)
  * avg_tokens   — answer verbosity (here, verbose answers tend to be LESS faithful)
  * avg_gen_s    — generation latency

Key-free for Groq candidates: local embeddings + Groq generation + a FIXED judge
(the same judge model scores every candidate, so the comparison is fair).

Usage:
  MSK_EMBED_PROVIDER=local MSK_CHROMA_DIR=<store> MSK_COLLECTION=msk_chunks \
    python scripts/model_bakeoff.py --max-cases 6 --key-file groq.key
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

# Candidates reachable with only a Groq key. Add other providers here once their
# server keys are configured — the harness handles any pinned provider/model.
DEFAULT_CANDIDATES: List[Dict[str, Any]] = [
    {"name": "gpt-oss-120b @2048", "provider": "groq", "model": "openai/gpt-oss-120b", "num_predict": 2048},
    {"name": "gpt-oss-20b @2048", "provider": "groq", "model": "openai/gpt-oss-20b", "num_predict": 2048},
    {"name": "gpt-oss-120b @700", "provider": "groq", "model": "openai/gpt-oss-120b", "num_predict": 700},
]


def load_questions(path: Path, limit: int) -> List[str]:
    qs = [json.loads(l)["question"] for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return qs[:limit] if limit else qs


def evidence_text(contexts) -> str:
    return "\n\n".join((c.get("text") or "") for c in (contexts or []))


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce per-question rows to the candidate's scorecard."""
    n = len(rows)
    scored = [r for r in rows if r["faithfulness"] is not None]
    llm = [r for r in rows if str(r["answer_mode"]).startswith("llm:")]
    refusals = [r for r in rows if r["faithfulness"] is None or str(r["answer_mode"]) == "evidence_only"]
    return {
        "n": n,
        "n_scored": len(scored),
        "faithfulness": (_mean([r["faithfulness"] for r in scored]) if scored else None),
        "refusal_rate": (len(refusals) / n) if n else 0.0,
        "llm_rate": (len(llm) / n) if n else 0.0,
        "avg_tokens": _mean([r["output_tokens"] for r in rows]),
        "avg_gen_s": _mean([r["gen_time"] for r in rows if r["gen_time"] is not None]),
    }


def format_table(results: Dict[str, Any]) -> str:
    ranked = sorted(results.values(),
                    key=lambda r: (r["agg"]["faithfulness"] if r["agg"]["faithfulness"] is not None else -1.0),
                    reverse=True)
    hdr = f"{'candidate':22} {'faith':>6} {'refuse':>7} {'llm%':>5} {'tok':>6} {'gen_s':>6}"
    lines = [hdr, "-" * len(hdr)]
    for r in ranked:
        a = r["agg"]
        f = a["faithfulness"]
        lines.append(f"{r['candidate']['name']:22} "
                     f"{('n/a' if f is None else f'{f:.2f}'):>6} "
                     f"{a['refusal_rate']*100:6.0f}% {a['llm_rate']*100:4.0f}% "
                     f"{a['avg_tokens']:6.0f} {a['avg_gen_s']:6.1f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--max-cases", type=int, default=6)
    p.add_argument("--judge-model", default="openai/gpt-oss-120b")
    p.add_argument("--key-file", type=Path, default=None)
    p.add_argument("--sleep", type=float, default=6.0, help="Seconds between calls (TPM throttle).")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    questions = load_questions(args.questions, args.max_cases)
    cands = DEFAULT_CANDIDATES
    print(f"Model bake-off: {len(cands)} candidates x {len(questions)} questions")
    if args.dry_run:
        for c in cands:
            print(f"  candidate: {c['name']}  (provider={c['provider']} num_predict={c['num_predict']})")
        print("DRY RUN — no API calls, no writes.")
        return 0

    import qaEngine as qa
    from openai import OpenAI

    key = (args.key_file.read_text().strip() if args.key_file
           else os.getenv("GROQ_API_KEY", "").strip())
    if not key:
        print("No Groq key. Set GROQ_API_KEY or pass --key-file.")
        return 2
    os.environ["GROQ_API_KEY"] = key
    judge = F.make_groq_judge(
        OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1"), args.judge_model)

    print(f"Backend: EMBED_PROVIDER={qa.EMBED_PROVIDER}  judge={args.judge_model}\n")

    results: Dict[str, Any] = {}
    for c in cands:
        cfg = qa.QAConfig(generation_provider=c["provider"], generation_model=c["model"],
                          num_predict=c["num_predict"])
        rows: List[Dict[str, Any]] = []
        print(f"--- {c['name']} ---")
        for i, q in enumerate(questions):
            try:
                res = qa.agentic_run(q, cfg=cfg)
            except Exception as exc:
                print(f"  [{i+1}/{len(questions)}] error: {type(exc).__name__}: {exc}")
                continue
            sc = F.score_answer(res.get("answer", ""), evidence_text(res.get("contexts")), judge=judge)
            rows.append({
                "q": q, "faithfulness": sc["score"], "n_claims": sc["n_claims"],
                "answer_mode": res.get("answer_mode"),
                "output_tokens": res.get("output_tokens") or 0,
                "gen_time": res.get("generation_time"),
            })
            s = sc["score"]
            print(f"  [{i+1}/{len(questions)}] faith={'n/a' if s is None else f'{s:.2f}'}  "
                  f"tok={res.get('output_tokens')}  mode={res.get('answer_mode')}")
            time.sleep(args.sleep)
        results[c["name"]] = {"candidate": c, "rows": rows, "agg": aggregate(rows)}

    print("\n=== BAKE-OFF RESULTS (ranked by faithfulness) ===")
    print(format_table(results))
    ranked = sorted(results.values(),
                    key=lambda r: (r["agg"]["faithfulness"] if r["agg"]["faithfulness"] is not None else -1.0),
                    reverse=True)
    print(f"\nHighest faithfulness: {ranked[0]['candidate']['name']}")
    print("(Weigh against refuse/llm%/latency for the final default — see the table.)")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
