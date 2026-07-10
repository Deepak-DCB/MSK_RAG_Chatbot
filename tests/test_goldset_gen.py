"""Zero-cost tests for the retrieval gold-set generator's pure logic.

Covers the question validator/normalizer (the quality gate) and prompt building.
No API calls, no corpus load.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import generate_retrieval_goldens as g


# ── clean_question: normalize + reject junk ──────────────────────────────────

def test_clean_question_accepts_good():
    q = g.clean_question("What causes atlantoaxial instability in the upper cervical spine?")
    assert q == "What causes atlantoaxial instability in the upper cervical spine?"


def test_clean_question_appends_missing_qmark():
    q = g.clean_question("How does scalene hypertonicity narrow the interscalene triangle")
    assert q.endswith("?")


def test_clean_question_strips_label_and_quotes():
    q = g.clean_question('Question: "Why does lumbar extension provoke pain?"')
    assert q == "Why does lumbar extension provoke pain?"


def test_clean_question_takes_first_line_only():
    q = g.clean_question("What is TOS?\nHere is some extra rambling the model added.")
    assert q == "What is TOS?"


def test_clean_question_rejects_meta_references():
    # Must not leak "this passage / the text / this article" phrasing.
    assert g.clean_question("According to the passage, what is APT?") is None
    assert g.clean_question("What does the above text say about POTS?") is None


def test_clean_question_rejects_too_short():
    assert g.clean_question("Why?") is None
    assert g.clean_question("Hip pain?") is None  # < 4 words


def test_clean_question_handles_empty():
    assert g.clean_question("") is None
    assert g.clean_question("   ") is None


# ── is_bad: same corpus filter as the store builders ─────────────────────────

def test_is_bad_matches_builder_filter():
    assert g.is_bad("") is True
    assert g.is_bad("too short") is True
    assert g.is_bad("See Figure 2") is True
    good = "The lumbar plexus arises from L1 to L4 and gives rise to several major nerves."
    assert g.is_bad(good) is False


# ── build_user_prompt: includes article + section header ─────────────────────

def test_build_user_prompt_includes_title_and_section():
    row = {"title": "Thoracic Outlet Syndrome", "section": "Mechanism", "body": "Scalene compression."}
    prompt = g.build_user_prompt(row, "body")
    assert "Thoracic Outlet Syndrome" in prompt
    assert "Mechanism" in prompt
    assert "Scalene compression." in prompt


def test_build_user_prompt_handles_missing_section():
    row = {"title": "T", "section": "", "body": "Body text here."}
    prompt = g.build_user_prompt(row, "body")
    assert "Section:" not in prompt  # omitted when blank
    assert "Body text here." in prompt
