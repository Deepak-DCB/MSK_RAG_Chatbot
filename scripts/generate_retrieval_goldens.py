#!/usr/bin/env python3
"""
generate_retrieval_goldens.py — build a retrieval gold set with a FREE provider.

The shipped datasets/retrieval-goldens.jsonl is too small (a couple of usable
queries) to judge retrieval quality. This grows it: for a stratified sample of
substantive corpus chunks, it asks a free LLM (Groq gpt-oss by default) to write
one realistic user question that the chunk answers, and records that chunk as the
ground truth. No OpenAI key required.

Each output line matches the existing golden schema so scripts/eval_retrieval_local.py
consumes it directly:
    {"gold_id", "question", "source_relpath", "gt_chunk_ids": [chunk_id],
     "article_id", "section"}

Quality notes / honest caveats:
  * Synthetic golds label the SOURCE chunk as ground truth — a known-relevant
    chunk, so Hit@k measures "did retrieval surface a known-relevant chunk". Other
    chunks may also be relevant; this is an approximate (single-positive) gold,
    same convention as the hand-written set.
  * The prompt forbids copying chunk phrasing / saying "this passage", so questions
    aren't trivially lexical. Still, spot-check before trusting the numbers.

Usage:
    # zero-cost preview (no API calls): which chunks, how many, est. tokens
    python scripts/generate_retrieval_goldens.py --dry-run --per-article 3

    # real run against Groq (reads key from GROQ_API_KEY or --key-file)
    GROQ_API_KEY=... python scripts/generate_retrieval_goldens.py --per-article 3 \
        --out datasets/retrieval-goldens-generated.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "MSKArticlesINDEX" / "chunks.parquet"
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "retrieval-goldens-generated.jsonl"

MIN_LEN = 50
CAPTION_RE = re.compile(r"\b(fig(?:ure)?|source|click|image|photo|credit)\b", re.I)
BAD_Q_RE = re.compile(r"\b(passage|excerpt|the (?:above )?text|this article|this section)\b", re.I)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You write realistic search questions for a musculoskeletal biomechanics and "
    "neurology Q&A system. Given a passage, output ONE natural question that a "
    "patient or clinician would type, which the passage answers. Rules: make it "
    "self-contained (never say 'this passage', 'the text', 'the article'); be "
    "specific to the passage's topic; phrase it in your own words (do NOT copy long "
    "phrases from the passage); one sentence ending in '?'. Output ONLY the question."
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("goldgen")


def is_bad(t: str) -> bool:
    return (not t) or len(t) < MIN_LEN or bool(CAPTION_RE.search(t))


def clean_question(raw: str) -> Optional[str]:
    """Normalize + validate a generated question; return None if unusable."""
    if not raw or not raw.strip():
        return None
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    q = lines[0]
    # Drop a leading "Question:" label, THEN strip surrounding quotes.
    q = re.sub(r"^\s*question\s*[:\-]\s*", "", q, flags=re.I).strip()
    q = q.strip('"').strip("'").strip()
    if len(q) < 12 or len(q.split()) < 3:
        return None
    if BAD_Q_RE.search(q):
        return None
    if not q.endswith("?"):
        q = q.rstrip(".") + "?"
    return q


def stratified_sample(df, per_article: int, seed: int):
    """Pick up to `per_article` substantive chunks from each article."""
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    textcol = "body" if "body" in df.columns else "embed_text"
    for _, group in df.groupby("article_id"):
        candidates = [r for r in group.to_dict("records")
                      if not is_bad(str(r.get(textcol) or ""))]
        rng.shuffle(candidates)
        rows.extend(candidates[:per_article])
    rng.shuffle(rows)
    return rows, textcol


def build_user_prompt(row: Dict[str, Any], textcol: str) -> str:
    title = str(row.get("title") or "").strip()
    section = str(row.get("section") or "").strip()
    body = str(row.get(textcol) or "").strip()
    head = f"[Article: {title}]" + (f" [Section: {section}]" if section else "")
    return f"{head}\n\n{body}"


def generate_one(client, model: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_prompt}],
        max_tokens=200,
        temperature=0.7,
        extra_body={"reasoning_effort": "low"},  # gpt-oss: keep reasoning cheap/reliable
    )
    return resp.choices[0].message.content or ""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-article", type=int, default=3, help="Chunks sampled per article.")
    p.add_argument("--max-chunks", type=int, default=0, help="Global cap (0 = no cap).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--sleep", type=float, default=4.0, help="Seconds between calls (TPM throttle).")
    p.add_argument("--key-file", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true", help="Sample + report only; no API calls.")
    args = p.parse_args(argv)

    import pandas as pd
    df = pd.read_parquet(CHUNKS_PATH)
    rows, textcol = stratified_sample(df, args.per_article, args.seed)
    if args.max_chunks > 0:
        rows = rows[:args.max_chunks]

    est_tokens = sum(min(700, len(str(r.get(textcol) or "")) // 4 + 200) for r in rows)
    log.info("Sampled %d chunks from %d articles (text column: %s). Est. ~%d input tokens.",
             len(rows), df["article_id"].nunique(), textcol, est_tokens)

    if args.dry_run:
        for r in rows[:5]:
            log.info("  would ask about: [%s] %s", str(r.get("title"))[:40], str(r.get("section"))[:40])
        log.info("DRY RUN — no API calls, no writes. Re-run without --dry-run to generate.")
        return 0

    # ── Live generation via Groq (OpenAI-compatible) ──
    import os
    from openai import OpenAI
    key = (args.key_file.read_text().strip() if args.key_file
           else os.getenv("GROQ_API_KEY", "").strip())
    if not key:
        log.error("No Groq key. Set GROQ_API_KEY or pass --key-file.")
        return 2
    client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = kept = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            prompt = build_user_prompt(r, textcol)
            q = None
            for attempt in range(4):
                try:
                    q = clean_question(generate_one(client, args.model, prompt))
                    break
                except Exception as exc:  # 429 / transient — back off and retry
                    wait = _retry_after(exc, default=args.sleep * (attempt + 2))
                    log.warning("  [%d/%d] %s — retry in %.1fs", i + 1, len(rows),
                                type(exc).__name__, wait)
                    time.sleep(wait)
            if not q:
                log.warning("  [%d/%d] no usable question; skipped", i + 1, len(rows))
                continue
            rec = {
                "gold_id": f"gen-{i+1:03d}",
                "question": q,
                "source_relpath": str(r.get("source_relpath") or ""),
                "gt_chunk_ids": [str(r.get("chunk_id") or "")],
                "article_id": str(r.get("article_id") or ""),
                "section": str(r.get("section") or ""),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            kept += 1
            if (i + 1) % 10 == 0:
                log.info("  progress %d/%d (kept %d)", i + 1, len(rows), kept)
            time.sleep(args.sleep)

    log.info("Wrote %d gold records to %s", written, args.out)
    return 0


def _retry_after(exc: Exception, default: float) -> float:
    m = re.search(r"try again in ([0-9.]+)s", str(exc))
    if m:
        return min(30.0, float(m.group(1)) + 0.5)
    return min(30.0, default)


if __name__ == "__main__":
    raise SystemExit(main())
