"""Zero-cost tests for the faithfulness (groundedness) scorer.

Covers claim decomposition, judge-prompt building, verdict parsing, and scoring
with a fake judge. No API calls.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import faithfulness as F


# ── split_claims ─────────────────────────────────────────────────────────────

def test_split_claims_splits_sentences():
    claims = F.split_claims("The scalenes compress the plexus. This causes tingling in the arm.")
    assert claims == ["The scalenes compress the plexus.",
                      "This causes tingling in the arm."]


def test_split_claims_drops_fragments_and_questions():
    claims = F.split_claims("Yes. Why does it hurt? The lumbar plexus arises from L1 to L4 roots.")
    # "Yes." (<4 words) and the question are dropped; the real claim remains.
    assert claims == ["The lumbar plexus arises from L1 to L4 roots."]


def test_split_claims_empty():
    assert F.split_claims("") == []
    assert F.split_claims("   ") == []


def test_split_claims_strips_markdown_structure():
    answer = (
        "## Mechanism of TOS\n"
        "**Key point:**\n"
        "- The scalenes can compress the brachial plexus in the interscalene triangle.\n"
        "1. This may produce tingling that radiates into the forearm.\n"
    )
    claims = F.split_claims(answer)
    # Heading and short bold label dropped; the two real prose claims survive, clean.
    assert claims == [
        "The scalenes can compress the brachial plexus in the interscalene triangle.",
        "This may produce tingling that radiates into the forearm.",
    ]


def test_split_claims_drops_no_evidence_meta():
    # An honest refusal must not be counted (or later penalized) as a false claim.
    assert F.split_claims("Insufficient evidence in the supplied context to answer.") == []
    assert F.split_claims("I am not able to answer that from the provided context.") == []


def test_split_claims_strips_label_prefix_keeps_claim():
    claims = F.split_claims("Answer: The subcostal nerve arises from the T12 spinal root.")
    assert claims == ["The subcostal nerve arises from the T12 spinal root."]


def test_split_claims_drops_preamble_and_list_leadins():
    answer = (
        "Below is a concise biomechanical explanation and practical guidance.\n"
        "Pathway: After exiting the spinal column it:\n"
        "The nerve then travels along the abdominal wall toward the midline.\n"
    )
    claims = F.split_claims(answer)
    # Framing sentence and the ':'-terminated lead-in are dropped; real claim kept.
    assert claims == ["The nerve then travels along the abdominal wall toward the midline."]


# ── parse_verdicts ───────────────────────────────────────────────────────────

def test_parse_verdicts_reads_lines():
    out = F.parse_verdicts("1: SUPPORTED\n2: UNSUPPORTED\n3: SUPPORTED", 3)
    assert out == [True, False, True]


def test_parse_verdicts_defaults_missing_to_false():
    # Only claim 1 judged; 2 and 3 missing -> conservative False.
    assert F.parse_verdicts("1: SUPPORTED", 3) == [True, False, False]


def test_parse_verdicts_tolerates_formatting():
    assert F.parse_verdicts("1) supported\n2. Unsupported", 2) == [True, False]


# ── score_answer ─────────────────────────────────────────────────────────────

def test_score_answer_no_judge_returns_none_score():
    r = F.score_answer("The plexus arises from L1 to L4 roots.", "evidence")
    assert r["score"] is None and r["n_claims"] == 1


def test_score_answer_no_claims_is_excluded_not_perfect():
    # A refusal / non-claim answer -> score None (excluded from aggregates), not 1.0.
    r = F.score_answer("Why?", "evidence", judge=lambda p: "")
    assert r["score"] is None and r["n_claims"] == 0


def test_score_answer_computes_fraction():
    answer = "The scalenes compress the plexus. Aliens built the pyramids here too."
    judge = lambda prompt: "1: SUPPORTED\n2: UNSUPPORTED"
    r = F.score_answer(answer, "The scalenes can compress the brachial plexus.", judge=judge)
    assert r["n_claims"] == 2
    assert r["n_supported"] == 1
    assert r["score"] == 0.5
    assert r["unsupported"] == ["Aliens built the pyramids here too."]


def test_build_judge_prompt_truncates_long_evidence():
    prompt = F.build_judge_prompt(["a claim here now"], "x" * 20000, max_evidence_chars=500)
    assert "EVIDENCE:" in prompt and "CLAIMS:" in prompt
    assert len(prompt) < 1200  # evidence capped
