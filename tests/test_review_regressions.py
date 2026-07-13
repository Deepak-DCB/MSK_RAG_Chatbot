"""Regressions for the bugs found in the deep code review.

Each test here pins a defect that shipped once and must not come back:

* red-flag escalation missed the lay phrasings people actually use
* the medication scope gate refused ordinary exercise questions
* the generation fallback ladder only degraded on key errors, so a transient
  OpenAI failure escaped the free-provider AND evidence-only paths as a hard 500
* a pinned provider that failed for any non-key reason did the same
* QAConfig shipped with the known-bad reranker ON by default
* a single malformed line disabled the entire concept graph
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "VectorDB"))

import graph_retrieval  # noqa: E402
import qaEngine  # noqa: E402
from qaEngine import QAConfig, detect_red_flags, detect_scope_issue  # noqa: E402


# ── Safety gate: lay phrasings of progressive neurologic deficit ──────────────

@pytest.mark.parametrize("prompt", [
    "my arm has been getting weaker every week",
    "I am getting weaker in both legs",
    "my legs give out when I walk",
    "my legs keep giving out",
    "the numbness in my feet is spreading up my legs",
    "I keep losing strength in my hand and dropping things",
    "I cant feel my legs properly",
    "my grip is failing and I drop cups",
])
def test_red_flag_catches_lay_phrasing_of_neurologic_deficit(prompt):
    """These all escaped the gate before: the patterns only knew the clinical
    vocabulary ("progressive weakness", "loss of strength"), not how people write."""
    assert "progressive_neurologic_deficit" in detect_red_flags(prompt)


@pytest.mark.parametrize("prompt", [
    "Numbness is stable but no weakness, no trauma, no fever, and no bowel or bladder changes.",
    "When should bowel or bladder changes with weakness be urgent?",
    "What are red flags for back pain?",
])
def test_red_flag_still_ignores_negated_and_general_info(prompt):
    """Broadening the gate must not break negation handling or turn general
    education questions into escalations."""
    assert detect_red_flags(prompt) == []


# ── Safety gate: standalone breathing symptoms ───────────────────────────────
#
# Every severe_chest_or_breathing pattern required chest pain to CO-OCCUR, so a bare
# "I am having trouble breathing" escaped the gate entirely. AGENTS.md already lists
# "severe chest pain or breathing symptoms" as an escalation criterion and the reason
# label already read "breathing symptoms" — this closed a code-vs-policy gap.

@pytest.mark.parametrize("prompt", [
    "I am having trouble breathing",
    "I have shortness of breath",
    "I cannot breathe properly",
    "I am short of breath",
    "I am struggling to breathe",
    "I cant catch my breath",
    "gasping for air",
    "I feel breathless",
])
def test_red_flag_catches_standalone_breathing_symptoms(prompt):
    assert "severe_chest_or_breathing" in detect_red_flags(prompt)


@pytest.mark.parametrize("prompt", [
    "severe chest pain",
    "chest pain and trouble breathing",
    "chest tightness with shortness of breath",
    "shortness of breath with chest pain",
])
def test_red_flag_still_catches_existing_chest_and_combined_cases(prompt):
    assert "severe_chest_or_breathing" in detect_red_flags(prompt)


@pytest.mark.parametrize("prompt", [
    # The corpus itself is ABOUT breathing mechanics — first-rib elevation, scalene
    # function, thoracic expansion. Matching the mechanism instead of the symptom would
    # escalate the product's own subject matter and make it useless.
    "How do breathing mechanics affect thoracic outlet syndrome?",
    "What breathing exercises help scalene function?",
    "Explain diaphragmatic breathing for rib mechanics",
    "Does poor breathing pattern cause first rib elevation failure?",
    # Negation and general-education controls must survive the broadening.
    "No trouble breathing, no chest pain, just neck stiffness.",
    "I have no shortness of breath at all.",
    "When should shortness of breath be urgent?",
])
def test_breathing_gate_does_not_over_escalate(prompt):
    assert detect_red_flags(prompt) == []


# ── Scope gate: medication boundary must not swallow exercise questions ───────

@pytest.mark.parametrize("prompt", [
    "Should I take a rest day after lifting?",
    "Should I take a break from running?",
    "My neck hurts, should I take it easy?",
])
def test_medication_gate_does_not_refuse_exercise_questions(prompt):
    """A bare `should i take` in the medication regex refused these as
    'medication advice' — core in-scope MSK questions the product exists to answer."""
    assert detect_scope_issue(prompt) is None


@pytest.mark.parametrize("prompt", [
    "Should I take ibuprofen for shoulder pain?",
    "Should I start steroids for sciatica?",
    "What dosage of naproxen should I use?",
    "Should I take something for the pain?",
])
def test_medication_gate_still_catches_real_medication_questions(prompt):
    assert detect_scope_issue(prompt) == "medication_advice"


# ── Config default ───────────────────────────────────────────────────────────

def test_reranker_is_off_by_default():
    """Ablation: Hit@5 collapses 94% -> 38% with the reranker on. A bare QAConfig()
    must inherit the safe default; it previously defaulted to True."""
    assert QAConfig().use_reranker is False


# ── Generation fallback ladder ───────────────────────────────────────────────

def _ctx():
    return [{"text": "Scapular depression narrows the costoclavicular space.",
             "meta": {"source_relpath": "a.html", "section": "Mechanism"}}]


def test_transient_openai_failure_falls_through_to_evidence_only(monkeypatch):
    """A non-key OpenAI error (transient 5xx, reset connection, rejected param) used to
    escape both the free-provider chain and the evidence-only answer as a hard 500."""
    def boom(*args, **kwargs):
        raise RuntimeError("Connection reset by peer")

    monkeypatch.setattr(qaEngine, "ask_openai_llm", boom)
    monkeypatch.setattr(qaEngine, "_configured_providers", lambda: [])

    text, _pt, _ot, mode, model = qaEngine.generate_answer_with_fallback(
        "prompt", QAConfig(), _ctx(), None, "why does my arm go numb?",
    )

    assert mode == "evidence_only"
    assert model is None
    assert "costoclavicular" in text


def test_pinned_provider_with_bad_model_falls_through_to_evidence_only(monkeypatch):
    """The UI accepts custom model strings. A bogus one raises NotFoundError (not an
    OpenAIKeyError), which used to propagate as a 500 instead of degrading."""
    def not_found(*args, **kwargs):
        raise RuntimeError("404 model_not_found: no such model 'gpt-nonexistent'")

    monkeypatch.setattr(qaEngine, "ask_openai_llm", not_found)
    cfg = QAConfig(generation_provider="openai", generation_model="gpt-nonexistent")

    text, _pt, _ot, mode, model = qaEngine.generate_answer_with_fallback(
        "prompt", cfg, _ctx(), None, "why does my arm go numb?",
    )

    assert mode == "evidence_only"
    assert model is None
    assert text


def test_midstream_failure_still_raises_and_does_not_duplicate_output(monkeypatch):
    """The guard that matters: once tokens have reached the client we must NOT restart
    on another provider, or the user sees two half-answers concatenated."""
    def fail_after_emitting(prompt, model, num_predict, on_token=None, **kwargs):
        if on_token:
            on_token("partial ")
        raise RuntimeError("stream died mid-flight")

    monkeypatch.setattr(qaEngine, "ask_openai_llm", fail_after_emitting)
    monkeypatch.setattr(qaEngine, "_configured_providers", lambda: [])

    emitted = []
    with pytest.raises(RuntimeError):
        qaEngine.generate_answer_with_fallback(
            "prompt", QAConfig(), _ctx(), None, "q", on_token=emitted.append,
        )
    assert emitted == ["partial "]


# ── Graph loader resilience ──────────────────────────────────────────────────

def test_one_malformed_line_does_not_disable_the_whole_graph(tmp_path):
    """A stray editor keystroke put `op` in front of line 1 of nodes.jsonl, which raised
    JSONDecodeError out of load_graph and silently turned the concept graph off
    everywhere. One bad line must cost one node, not the graph."""
    good = {"node_id": "n1", "canonical_name": "brachial plexus", "node_type": "nerve"}
    other = {"node_id": "n2", "canonical_name": "scalene", "node_type": "muscle"}

    (tmp_path / "nodes.jsonl").write_text(
        "op" + json.dumps(good) + "\n" + json.dumps(other) + "\n", encoding="utf-8"
    )
    for name in ("edges.jsonl", "paths.jsonl", "claims.jsonl"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "graph_manifest.json").write_text("{}", encoding="utf-8")

    graph = graph_retrieval.load_graph(tmp_path)

    assert graph["available"] is True
    assert [n["node_id"] for n in graph["nodes"]] == ["n2"]  # bad line skipped, not fatal


def test_valid_json_that_is_not_an_object_is_skipped_not_fatal(tmp_path):
    """A bare string/number/list/null parses as valid JSON, so it used to survive into
    `nodes` and then raise AttributeError on node.get(...) inside load_graph — OUTSIDE
    its try/except — taking the whole graph down by a route the JSONDecodeError guard
    never covered."""
    good = {"node_id": "n1", "canonical_name": "brachial plexus", "node_type": "nerve"}

    (tmp_path / "nodes.jsonl").write_text(
        "\n".join([
            '"just a string"',   # valid JSON, not an object
            "123",               # valid JSON, not an object
            "[1, 2, 3]",         # valid JSON, not an object
            "null",              # valid JSON, not an object
            json.dumps(good),
        ]) + "\n",
        encoding="utf-8",
    )
    for name in ("edges.jsonl", "paths.jsonl", "claims.jsonl"):
        (tmp_path / name).write_text("", encoding="utf-8")
    (tmp_path / "graph_manifest.json").write_text("{}", encoding="utf-8")

    graph = graph_retrieval.load_graph(tmp_path)

    assert graph["available"] is True, "non-object rows must not disable the graph"
    assert [n["node_id"] for n in graph["nodes"]] == ["n1"]


def test_real_concept_graph_loads_and_finds_mechanism_paths():
    """Guards the actual committed artifact: it was corrupt on disk and nobody noticed
    because every failure path degraded silently."""
    graph = graph_retrieval.load_graph()
    if not graph.get("available"):
        pytest.skip(f"graph artifacts unavailable: {graph.get('fallback_reason')}")

    pack = graph_retrieval.build_graph_context(
        "scapular depression causing thoracic outlet numbness"
    )
    assert pack["available"] is True
    assert pack["paths"], "expected at least one mechanism path for a core corpus query"
