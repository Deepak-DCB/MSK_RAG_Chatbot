"""Phase 1: typed OpenAI quota/auth error classification.

These tests pin the contract that quota/auth/invalid-key failures are surfaced as
`OpenAIKeyError` (code ``api_key_unavailable``) while transient rate-limits and
unrelated errors are left alone, so the pipeline can distinguish "the key is dead"
from "something else went wrong".
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "VectorDB"))

import qaEngine  # noqa: E402


def test_insufficient_quota_string_is_key_error():
    exc = Exception("Error code: 429 - insufficient_quota: exceeded your current quota")
    assert qaEngine._classify_openai_error(exc) == "api_key_unavailable"


def test_transient_rate_limit_is_not_key_error():
    exc = Exception("Error code: 429 - rate limit reached, please retry shortly")
    assert qaEngine._classify_openai_error(exc) is None


def test_invalid_api_key_string_is_key_error():
    assert qaEngine._classify_openai_error(Exception("Incorrect API key provided")) == "api_key_unavailable"


def test_unrelated_error_is_not_key_error():
    assert qaEngine._classify_openai_error(Exception("connection reset by peer")) is None


def test_openai_key_error_passthrough_preserves_code():
    err = qaEngine.OpenAIKeyError("boom", code="api_key_unavailable")
    assert err.code == "api_key_unavailable"
    assert qaEngine._classify_openai_error(err) == "api_key_unavailable"


def test_sdk_authentication_error_is_key_error():
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "http://example.test")
    response = httpx.Response(401, request=request)
    exc = openai.AuthenticationError("bad key", response=response, body=None)
    assert qaEngine._classify_openai_error(exc) == "api_key_unavailable"


def test_missing_env_key_raises_key_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Force a fresh client construction so the missing-key branch is exercised.
    monkeypatch.setattr(qaEngine, "_openai_client", None, raising=False)
    with pytest.raises(qaEngine.OpenAIKeyError):
        qaEngine._get_openai_client()


# ── Phase 2: classify/rewrite degrade instead of crashing ────────────────────

def test_classify_rewrite_degrade_when_utility_model_is_dead(monkeypatch):
    """When the utility model is out of quota, agentic_run must still reach
    retrieval with a broad category + the original question, not crash."""
    def _boom(*args, **kwargs):
        raise qaEngine.OpenAIKeyError("dead")

    captured = {}

    def _fake_run_qa(refined_q, **kwargs):
        captured["refined_q"] = refined_q
        return {"answer": "stub", "citations": []}

    monkeypatch.setattr(qaEngine, "classify_query", _boom)
    monkeypatch.setattr(qaEngine, "rewrite_query", _boom)
    monkeypatch.setattr(qaEngine, "run_qa", _fake_run_qa)
    # Skip the preflight so we exercise the classify/rewrite path directly.
    monkeypatch.setattr(qaEngine, "local_preflight", lambda q, history=None: {"action": "continue"})

    res = qaEngine.agentic_run("why does my shoulder ache?")

    assert res["category"] == "D"
    assert res["query_processing_degraded"] is True
    assert captured["refined_q"] == "why does my shoulder ache?"  # original question used


# ── Phase 3: BM25-only retrieval fallback ────────────────────────────────────

class _FakeCollection:
    def __init__(self, docs, metas):
        self._docs = docs
        self._metas = metas
        self._ids = [str(i) for i in range(len(docs))]

    def count(self):
        return len(self._docs)

    def get(self, include=None):
        return {"ids": self._ids, "documents": self._docs, "metadatas": self._metas}

    def query(self, **kwargs):  # pragma: no cover - must not be reached when dense fails
        raise AssertionError("dense query should not run when embeddings are unavailable")


def test_bm25_only_result_shape_and_normalized_distances():
    hits = [
        {"text": "shoulder impingement mechanics", "meta": {"source_relpath": "a"}, "bm25_score": 4.0},
        {"text": "scapular dyskinesis", "meta": {"source_relpath": "b"}, "bm25_score": 2.0},
    ]
    out = qaEngine._bm25_only_result(hits, retrieval_pool=10)
    assert out["documents"] == [["shoulder impingement mechanics", "scapular dyskinesis"]]
    assert out["metadatas"] == [[{"source_relpath": "a"}, {"source_relpath": "b"}]]
    dists = out["distances"][0]
    # top hit -> 0.5 (best), all within [0.5, 1.0], monotonic non-decreasing
    assert dists[0] == pytest.approx(0.5)
    assert all(0.5 <= d <= 1.0 for d in dists)
    assert dists[0] <= dists[1]


def test_bm25_only_result_empty():
    out = qaEngine._bm25_only_result([], retrieval_pool=10)
    assert out == {"documents": [[]], "metadatas": [[]], "distances": [[]]}


def test_hybrid_search_falls_back_to_bm25_when_embeddings_fail(monkeypatch):
    pytest.importorskip("rank_bm25")
    docs = [
        "thoracic outlet syndrome brachial plexus compression",
        "levator scapulae postural strain neck pain",
        "median nerve entrapment carpal tunnel wrist",
    ]
    metas = [{"source_relpath": f"doc{i}"} for i in range(len(docs))]
    coll = _FakeCollection(docs, metas)

    # Force a fresh BM25 index and a dead embedding path.
    monkeypatch.setattr(qaEngine.BM25Index, "_instance", None, raising=False)
    def _dead_embed(text):
        raise qaEngine.OpenAIKeyError("dead")
    monkeypatch.setattr(qaEngine, "encode_query", _dead_embed)

    qaEngine._retrieval_degraded.set(False)
    raw = qaEngine.hybrid_search("thoracic outlet brachial plexus", coll, retrieval_pool=10)

    assert qaEngine._retrieval_degraded.get() is True
    docs_out = raw["documents"][0]
    assert docs_out, "expected BM25 keyword hits"
    assert any("brachial plexus" in d for d in docs_out)


# ── Phase 4: deterministic evidence-only answer ──────────────────────────────

class _Pack:
    def __init__(self, spans):
        self.selected_evidence_spans = spans


def test_evidence_only_answer_is_grounded_and_conservative():
    spans = [
        {
            "source_relpath": "mskneurology.com/tos",
            "section_name": "Thoracic outlet syndrome",
            "title": "TOS",
            "text": "Costoclavicular narrowing can compress the brachial plexus and produce neuralgia.",
        }
    ]
    out = qaEngine.build_evidence_only_answer([], _Pack(spans), "why does my arm tingle?")
    # grounded: shows the source and the retrieved text
    assert "mskneurology.com/tos" in out
    assert "brachial plexus" in out
    # honest about the degraded mode + conservative seek-care framing
    assert "temporarily unavailable" in out.lower()
    assert "in-person" in out.lower()
    assert "not a\ndiagnosis" in out or "not a diagnosis" in out.replace("\n", " ")
    # must NOT invent reassurance
    lowered = out.lower()
    assert "likely benign" not in lowered
    assert "nothing to worry" not in lowered


def test_evidence_only_answer_falls_back_to_raw_context_when_no_spans():
    context = [{"text": "Levator scapulae strain from sustained posture.", "meta": {"source_relpath": "doc1", "section": "Muscular"}}]
    out = qaEngine.build_evidence_only_answer(context, _Pack([]), "neck ache")
    assert "Levator scapulae" in out
    assert "doc1" in out


def test_evidence_only_answer_handles_empty_evidence():
    out = qaEngine.build_evidence_only_answer([], _Pack([]), "q")
    assert "No supporting passages" in out
    assert "in-person" in out.lower()  # disclaimer still present
