from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import backend.main as main
import mechanics_retrieval
from scripts.eval_mechanics_study import evaluate_case

from conftest import requires_mechanics_artifacts


def make_client(monkeypatch) -> TestClient:
    monkeypatch.setattr(main, "_check_rate_limit", lambda ip: None)
    return TestClient(main.app)


@requires_mechanics_artifacts
def test_mechanics_study_endpoint_returns_structured_answer(monkeypatch):
    with make_client(monkeypatch) as client:
        response = client.post(
            "/study/mechanics",
            json={"question": "Where can the brachial plexus get compressed according to the current corpus?"},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["mechanics_available"] is True
    assert body["mechanics_fallback_reason"] is None
    assert body["mechanics_entrapment_sites"]
    assert body["mechanics_spaces"]
    assert body["mechanics_evidence_spans"]
    assert "**Short answer**" in body["answer"]
    assert "**Directly supported claims**" in body["answer"]
    assert "**Indirect or uncertain links**" in body["answer"]
    assert "**What the corpus does not prove**" in body["answer"]
    assert "diagnose" in body["answer"].lower()
    assert "you have" not in body["answer"].lower()


def test_mechanics_study_endpoint_safety_gate_short_circuits(monkeypatch):
    with make_client(monkeypatch) as client:
        response = client.post(
            "/study/mechanics",
            json={"question": "I have worsening arm weakness and new bowel and bladder changes."},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["mechanics_available"] is False
    assert body["mechanics_fallback_reason"] == "safety_gate_triggered"
    assert body["safety_gate_triggered"] is True
    assert "urgent in-person medical evaluation" in body["answer"]


def test_mechanics_study_endpoint_missing_artifacts_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        main,
        "build_mechanics_context",
        lambda question, max_items=8: mechanics_retrieval.build_mechanics_context(
            question,
            max_items=max_items,
            base_dir=tmp_path / "missing",
        ),
    )
    with make_client(monkeypatch) as client:
        response = client.post("/study/mechanics", json={"question": "How do scalenes and trapezius work together?"})

    body = response.json()
    assert response.status_code == 200
    assert body["mechanics_available"] is False
    assert body["mechanics_fallback_reason"] == "mechanics_artifacts_missing"
    assert "not available" in body["answer"].lower()


@requires_mechanics_artifacts
def test_mechanics_study_eval_case_passes():
    result = evaluate_case(
        {
            "id": "unit_scalenes_first_rib",
            "question": "What is the chain from scalenes to first rib to costoclavicular compression?",
            "expected_terms": ["scalenes", "first rib", "costoclavicular", "brachial plexus"],
            "expect_chain": True,
        }
    )

    assert result["passed"] is True
