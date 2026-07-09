from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTORDB_PATH = PROJECT_ROOT / "VectorDB"
if str(VECTORDB_PATH) not in sys.path:
    sys.path.insert(0, str(VECTORDB_PATH))

import qaEngine
from qaEngine import QAConfig, agentic_run, detect_red_flags, detect_scope_issue, local_preflight


def test_red_flag_gate_short_circuits_before_llm(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM classification should not run for red-flag prompts")

    monkeypatch.setattr(qaEngine, "classify_query", fail_if_called)
    streamed = []

    res = agentic_run(
        "I have worsening arm weakness this week and trouble controlling my bladder.",
        cfg=QAConfig(generate_answer=False),
        on_token=streamed.append,
    )

    assert res["safety_gate_triggered"] is True
    assert res["triage_level"] == "urgent_in_person_evaluation"
    assert "progressive_neurologic_deficit" in res["safety_gate_reasons"]
    assert "bowel_bladder" in res["safety_gate_reasons"]
    assert "urgent in-person medical evaluation" in res["answer"]
    assert streamed == [res["answer"]]
    assert res["prompt_tokens"] == 0


def test_red_flag_detector_respects_negated_controls():
    reasons = detect_red_flags(
        "Numbness is stable but no weakness, no trauma, no fever, and no bowel or bladder changes."
    )

    assert reasons == []


def test_red_flag_detector_catches_bowel_and_bladder_conjunction():
    reasons = detect_red_flags("I have new bowel and bladder changes with low back pain.")

    assert "bowel_bladder" in reasons


def test_red_flag_detector_does_not_gate_general_education_question():
    reasons = detect_red_flags("When should bowel or bladder changes with weakness be urgent?")

    assert reasons == []


def test_scope_boundary_catches_medication_and_diagnosis_requests():
    assert detect_scope_issue("Should I take ibuprofen for shoulder pain?") == "medication_advice"
    assert detect_scope_issue("Do I have thoracic outlet syndrome?") == "diagnosis_request"
    assert detect_scope_issue("Can you diagnose it?") == "diagnosis_request"
    assert detect_scope_issue("Should I start steroids for sciatica?") == "medication_advice"
    assert detect_scope_issue("Can you help me understand if this might be dangerous?") is None


def test_local_preflight_skips_clarification_for_followup_history():
    preflight = local_preflight(
        "What exercises help?",
        history=[
            {"role": "user", "content": "My arm tingles when carrying a bag."},
            {"role": "assistant", "content": "This may involve shoulder and thoracic outlet loading."},
        ],
    )

    assert preflight["action"] == "continue"


def test_vague_query_asks_compact_adaptive_clarification(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM classification should not run for vague prompts")

    monkeypatch.setattr(qaEngine, "classify_query", fail_if_called)

    res = agentic_run("Pain", cfg=QAConfig(generate_answer=False))

    assert res["category"] == "clarification"
    assert res["triage_level"] == "needs_more_detail"
    assert res["safety_gate_triggered"] is False
    assert "location" in res["answer"].lower()
    assert "sensation" in res["answer"].lower()
    assert "trigger/timing" in res["answer"].lower()
    assert "bowel/bladder changes" in res["answer"].lower()
    assert res["output_tokens"] > 0


def test_single_word_symptom_gets_clarification(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("LLM classification should not run for vague prompts")

    monkeypatch.setattr(qaEngine, "classify_query", fail_if_called)

    res = agentic_run("Headache", cfg=QAConfig(generate_answer=False))

    assert res["category"] == "clarification"


def test_low_confidence_fallback_short_circuits_generation(monkeypatch):
    def fake_hybrid_search(question, coll, retrieval_pool=50):
        return {
            "documents": [["Generic mention of pain without useful context."]],
            "metadatas": [[{"source_relpath": "misc/source.html", "section": "misc"}]],
            "distances": [[0.96]],
        }

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Answer generation should not run during low-confidence fallback")

    monkeypatch.setattr(qaEngine._backend, "load_collection", lambda: object())
    monkeypatch.setattr(qaEngine, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(qaEngine, "classify_query", lambda *_args, **_kwargs: "B")
    monkeypatch.setattr(qaEngine, "rewrite_query", lambda q, *_args, **_kwargs: q)
    monkeypatch.setattr(qaEngine, "ask_openai_llm", fail_if_called)

    res = agentic_run(
        "My shoulder is strange with no clear pattern yet.",
        cfg=QAConfig(
            generate_answer=True,
            use_reranker=False,
            low_confidence_fallback_threshold=0.99,
        ),
    )

    assert res.get("low_confidence_fallback") is True
    assert res["triage_level"] == "needs_more_detail"
    assert res["retrieval_confidence"] < 0.99
    assert "location" in res["answer"].lower()
    assert "urgent in-person evaluation" in res["answer"].lower()
