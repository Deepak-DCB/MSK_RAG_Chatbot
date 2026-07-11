#!/usr/bin/env python3
"""
faithfulness.py — groundedness scoring for RAG answers.

Makes "is the answer true to its evidence?" a measurable number instead of a hope.
Splits an answer into claim-sized sentences, asks an LLM judge whether each is
supported by the retrieved evidence, and returns a groundedness score in [0,1]
plus the list of unsupported claims. The judge runs on any OpenAI-compatible
provider (Groq by default) — no OpenAI key required.

Measurement only: this module reads answers, it never rewrites them. A runtime
faithfulness gate (flag/strip unsupported claims) is a separate, eval-gated step.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_VERDICT_RE = re.compile(r"(\d+)\s*[:.\)]\s*(SUPPORTED|UNSUPPORTED|PARTIAL)", re.I)

_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")           # '## Heading'
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")   # '- ', '1. ', '2) '
_MD_INLINE_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")   # [text](url) -> text
_MD_EMPHASIS = re.compile(r"[*_`]{1,3}")                 # **bold** _em_ `code`
# Meta / refusal statements are honest non-claims — they assert nothing about the
# world, so they must NOT be counted (or penalized) as factual claims.
_META_RE = re.compile(
    r"\b(insufficient evidence|not able to answer|cannot answer|no (?:relevant )?"
    r"(?:context|information|evidence)|based on the (?:supplied|provided) context|"
    r"according to the (?:supplied|provided) context)\b", re.I)
# Framing / preamble sentences assert nothing checkable ("Below is a summary…").
_LEADIN_RE = re.compile(r"^(below is|here('?s| is| are)|in summary|to summari[sz]e|"
                        r"the following|this (?:answer|response))\b", re.I)
# A short label prefix on a claim ("Answer:", "Pathway:", "Key point:") — strip it,
# keep the claim that follows.
_LABEL_PREFIX = re.compile(
    r"^(answer|pathway|note|summary|overview|mechanism|explanation|key point|"
    r"example|background|conclusion|result)s?\s*:\s*", re.I)

JUDGE_SYSTEM = (
    "You are a strict fact-checker for a medical Q&A system. You are given EVIDENCE "
    "and numbered CLAIMS taken from an answer. For each claim, decide whether it is "
    "directly supported by the EVIDENCE. A claim is UNSUPPORTED if it introduces "
    "facts, numbers, mechanisms, or conclusions not present in the EVIDENCE, even if "
    "they sound plausible. Reply with exactly one line per claim: 'N: SUPPORTED' or "
    "'N: UNSUPPORTED'. Output only those lines, nothing else."
)


def split_claims(answer: str) -> List[str]:
    """Decompose an answer into claim-sized sentences worth fact-checking.

    Markdown-aware: LLMs answer in markdown (headings, bold, numbered lists), and
    naive sentence splitting shatters that structure into formatting fragments that
    pollute the score. So we go line-by-line, drop headings, strip list markers and
    inline emphasis/links, then split prose into sentences. Fragments (<4 words),
    rhetorical questions, and honest 'no evidence' meta-statements are dropped —
    none carry a checkable factual assertion."""
    text = (answer or "").strip()
    if not text:
        return []
    claims: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or _MD_HEADING.match(line):     # blank or markdown heading
            continue
        line = _LIST_MARKER.sub("", line)           # drop bullet / number
        line = _MD_INLINE_LINK.sub(r"\1", line)     # links -> anchor text
        line = _MD_EMPHASIS.sub("", line).strip()   # bold/italic/code markers
        if not line:
            continue
        for s in _SENT_SPLIT.split(line):
            s = s.strip().strip("#").strip()
            s = _LABEL_PREFIX.sub("", s).strip()      # 'Answer: X' -> 'X'
            # Skip fragments, questions, and list lead-ins ('... it:').
            if len(s.split()) < 4 or s.endswith("?") or s.endswith(":"):
                continue
            if _META_RE.search(s) or _LEADIN_RE.match(s):
                continue
            claims.append(s)
    return claims


def build_judge_prompt(claims: List[str], evidence: str, *, max_evidence_chars: int = 8000) -> str:
    ev = (evidence or "").strip()
    if len(ev) > max_evidence_chars:
        ev = ev[:max_evidence_chars] + " …"
    lines = ["EVIDENCE:", ev, "", "CLAIMS:"]
    lines += [f"{i}. {c}" for i, c in enumerate(claims, 1)]
    return "\n".join(lines)


def parse_verdicts(text: str, n: int) -> List[bool]:
    """Parse 'N: SUPPORTED/UNSUPPORTED' judge lines into n booleans.

    Missing/unparseable verdicts default to False (unsupported) — a conservative
    choice for a safety-sensitive metric: unproven claims don't get credit."""
    verdicts = [False] * n
    for m in _VERDICT_RE.finditer(text or ""):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            verdicts[idx] = m.group(2).upper() == "SUPPORTED"
    return verdicts


def score_answer(answer: str, evidence: str, *, judge: Optional[Callable[[str], str]] = None) -> Dict[str, Any]:
    """Score one answer's faithfulness against its evidence.

    `judge` is a callable(prompt) -> str wrapping the LLM. If None, only the claim
    decomposition is returned (score None) — useful for tests and dry-runs.

    An answer with no checkable claims (e.g. a refusal) returns score None, not 1.0,
    so it is EXCLUDED from aggregates rather than counted as perfectly grounded."""
    claims = split_claims(answer)
    if not claims:
        return {"n_claims": 0, "n_supported": 0, "score": None, "unsupported": [], "claims": []}
    if judge is None:
        return {"n_claims": len(claims), "n_supported": 0, "score": None,
                "unsupported": [], "claims": claims}
    verdicts = parse_verdicts(judge(build_judge_prompt(claims, evidence)), len(claims))
    n_ok = sum(verdicts)
    return {
        "n_claims": len(claims),
        "n_supported": n_ok,
        "score": n_ok / len(claims),
        "unsupported": [c for c, v in zip(claims, verdicts) if not v],
        "claims": claims,
    }


def make_groq_judge(client, model: str, *, retries: int = 3, base_sleep: float = 4.0) -> Callable[[str], str]:
    """Build a judge callable backed by an OpenAI-compatible client (e.g. Groq).

    Retries on transient failures (429 rate limits) with a back-off parsed from the
    error when available, so a rate limit does not silently zero out a score."""
    def _judge(prompt: str) -> str:
        last = None
        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": prompt}],
                    max_tokens=400, temperature=0.0,
                    extra_body={"reasoning_effort": "low"},
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:  # transient / 429
                last = exc
                m = re.search(r"try again in ([0-9.]+)s", str(exc))
                time.sleep(min(30.0, float(m.group(1)) + 0.5 if m else base_sleep * (attempt + 1)))
        raise RuntimeError(f"judge failed after {retries} attempts: {last}")
    return _judge
