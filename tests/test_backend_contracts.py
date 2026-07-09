from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import backend.main as main


class FakeCollection:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


def make_client(monkeypatch, *, collection_count: int = 1301, agentic_run=None) -> TestClient:
    fake_collection = FakeCollection(collection_count)
    monkeypatch.setattr(main._backend, "collection", fake_collection, raising=False)
    monkeypatch.setattr(main._backend, "load_collection", lambda: fake_collection)
    monkeypatch.setattr(main, "_check_rate_limit", lambda ip: None)
    if agentic_run is not None:
        monkeypatch.setattr(main, "agentic_run", agentic_run)
    return TestClient(main.app)


def test_health_reports_collection_readiness(monkeypatch):
    with make_client(monkeypatch, collection_count=1301) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "chroma_loaded": True,
        "chunk_count": 1301,
    }


def test_ask_clamps_public_config_and_returns_metadata(monkeypatch):
    seen = {}

    def fake_agentic_run(question, cfg, history=None, on_token=None):
        seen["question"] = question
        seen["use_reranker"] = cfg.use_reranker
        seen["reranker_top_n"] = cfg.reranker_top_n
        seen["history"] = history
        return {
            "answer": "Conservative answer grounded in retrieved context.",
            "citations": ["mskneurology.com/how-truly-treat-thoracic-outlet-syndrome/index.html"],
            "retrieval_confidence": 0.61,
            "retrieval_time": 0.42,
            "generation_time": 1.17,
            "prompt_tokens": 150,
            "output_tokens": 75,
            "context_tokens": 120,
            "question_tokens": 10,
            "category": "structured_biomechanical_pattern",
            "category_label": "Structured biomechanical pattern",
            "refined_query": "thoracic outlet symptoms with scapular depression",
        }

    with make_client(monkeypatch, agentic_run=fake_agentic_run) as client:
        response = client.post(
            "/ask",
            json={
                "question": "Could this be thoracic outlet related?",
                "history": [{"role": "user", "content": "My shoulder drops."}],
                "config": {
                    "use_reranker": True,
                    "reranker_top_n": 999,
                    "retrieval_pool": 500,
                },
            },
        )

    body = response.json()
    assert response.status_code == 200
    assert seen == {
        "question": "Could this be thoracic outlet related?",
        "use_reranker": True,
        "reranker_top_n": 10,
        "history": [{"role": "user", "content": "My shoulder drops."}],
    }
    assert body["reranker_mode"] == "per_source"
    assert body["config_source"] == "request_override"
    assert body["reranker_top_n"] == 10
    assert body["refined_query"] == "thoracic outlet symptoms with scapular depression"
    assert body["safety_gate_triggered"] is False
    assert body["safety_gate_reasons"] == []


def test_stream_done_event_includes_telemetry(monkeypatch):
    def fake_agentic_run(question, cfg, history=None, on_token=None):
        assert cfg.use_reranker is False
        if on_token is not None:
            on_token("Grounded ")
            on_token("answer")
        return {
            "answer": "Grounded answer",
            "citations": ["mskneurology.com/example-source/index.html"],
            "retrieval_confidence": 0.5,
            "retrieval_time": 0.25,
            "generation_time": 0.75,
            "prompt_tokens": 100,
            "output_tokens": 20,
            "context_tokens": 80,
            "question_tokens": 8,
            "category": "structured_biomechanical_pattern",
            "category_label": "Structured biomechanical pattern",
            "refined_query": "rewritten query",
        }

    with make_client(monkeypatch, agentic_run=fake_agentic_run) as client:
        response = client.post(
            "/ask/stream",
            json={"question": "What structures are usually responsible?", "config": {"use_reranker": False}},
        )

    assert response.status_code == 200
    assert "event: done" in response.text
    assert '"complete": true' in response.text
    assert '"reranker_mode": "off"' in response.text
    assert '"refined_query": "rewritten query"' in response.text
    assert '"safety_gate_triggered": false' in response.text
