"""Streaming worker lifecycle: a stalled provider must not leak a generation thread.

Before this, the SSE generator returned on timeout while the daemon worker kept running
agentic_run() to completion — still calling on_token into a queue nobody read, and still
billing the provider. One stalled provider leaked one live thread + one live API request
per timeout, and user retries multiplied it.

Cancellation is cooperative: on_token is the only hook the provider architecture calls on
every streamed chunk, so it is the cancellation point. These tests use a *blocked* fake
provider to prove the worker actually winds down.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "VectorDB"))

from backend import main as backend_main  # noqa: E402


def _threads_named_alive() -> int:
    return sum(1 for t in threading.enumerate() if t.is_alive())


def _parse_done(body: str) -> dict:
    """Pull the JSON payload of the SSE `done` event out of a raw stream body."""
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("event: done"):
            for nxt in lines[i + 1:]:
                if nxt.startswith("data: "):
                    return json.loads(nxt[6:])
    raise AssertionError(f"no done event in stream body:\n{body}")


@pytest.fixture
def fast_deadlines(monkeypatch):
    """Shrink the deadlines so the stall path is exercised in ~1s, not 120s."""
    monkeypatch.setattr(backend_main, "STREAM_IDLE_TIMEOUT", 1)
    monkeypatch.setattr(backend_main, "STREAM_TOTAL_TIMEOUT", 2)
    monkeypatch.setattr(backend_main, "STREAM_JOIN_TIMEOUT", 5)


def test_slow_provider_past_deadline_is_cancelled_and_worker_does_not_leak(monkeypatch, fast_deadlines):
    """A provider that trickles tokens past the stream deadline must be ABANDONED.

    This is the leak from the review: the generator returned at the deadline while the
    worker kept generating (and billing) into a queue nobody read. Cancellation lands on
    the next on_token, so the worker unwinds instead of running to completion.
    """
    entered = threading.Event()
    worker_exited = threading.Event()
    finished_naturally = threading.Event()
    tokens_emitted = []

    def trickling_agentic_run(question, cfg=None, history=None, on_token=None,
                              conversation_summary=None):
        try:
            entered.set()
            # Slow-but-alive: keeps producing well past the 2s total deadline. Without
            # cancellation this would run for ~20s after the client is gone.
            for i in range(100):
                if on_token:
                    on_token(f"tok{i} ")      # raises _StreamCancelled once consumer is gone
                tokens_emitted.append(i)
                threading.Event().wait(0.2)
            finished_naturally.set()
            return {"answer": "should never be delivered", "answer_mode": "llm:openai"}
        finally:
            worker_exited.set()

    monkeypatch.setattr(backend_main, "agentic_run", trickling_agentic_run)

    with TestClient(backend_main.app) as client:
        r = client.post("/ask/stream", json={"question": "why does my arm go numb?"})
        assert r.status_code == 200
        body = r.text

    assert entered.is_set(), "the fake provider should have been invoked"

    # Streaming contract on a stream that blew its deadline.
    meta = _parse_done(body)
    assert meta["complete"] is False, "a timed-out stream must report complete: false"
    assert meta.get("error"), "a timed-out stream must carry an error"
    assert meta.get("request_id"), "a timed-out stream must carry a request_id"

    # Tokens produced before the deadline still reached the client incrementally.
    assert '"token"' in body, "tokens must stream incrementally before the deadline"

    # The point of the fix: the worker unwound instead of running to completion.
    assert worker_exited.wait(timeout=10), (
        "generation worker leaked: still running after the client was served"
    )
    assert not finished_naturally.is_set(), (
        "the worker generated to completion for a client that had already gone away"
    )
    assert len(tokens_emitted) < 100, "generation should have been cut short, not completed"


def test_cancellation_is_not_swallowed_as_a_provider_failure():
    """_StreamCancelled must subclass BaseException.

    If it subclassed Exception, qaEngine's broad `except Exception` in
    generate_answer_with_fallback() would catch it and kick off the fallback chain —
    starting a NEW generation for a client that already disconnected, which is the exact
    leak this mechanism exists to close.
    """
    assert issubclass(backend_main._StreamCancelled, BaseException)
    assert not issubclass(backend_main._StreamCancelled, Exception)


def test_successful_stream_still_completes_and_joins_cleanly(monkeypatch, fast_deadlines):
    """The reaper must not damage the happy path: tokens stream, complete: true, telemetry
    intact, and the worker exits on its own."""
    def quick_run(question, cfg=None, history=None, on_token=None, conversation_summary=None):
        for t in ["Scapular ", "depression ", "narrows the space."]:
            if on_token:
                on_token(t)
        return {
            "answer": "Scapular depression narrows the space.",
            "answer_mode": "llm:openai",
            "generation_model": "gpt-5.4-mini",
            "citations": ["mskneurology.com/tos — Mechanism"],
            "retrieval_confidence": 0.44,
            "retrieval_mode": "hybrid",
        }

    monkeypatch.setattr(backend_main, "agentic_run", quick_run)

    before = _threads_named_alive()
    with TestClient(backend_main.app) as client:
        r = client.post("/ask/stream", json={"question": "why does my arm go numb?"})
        assert r.status_code == 200
        body = r.text

    meta = _parse_done(body)
    assert meta["complete"] is True
    assert meta["citations"] == ["mskneurology.com/tos — Mechanism"]
    assert meta["generation_model"] == "gpt-5.4-mini"
    assert meta["retrieval_mode"] == "hybrid"
    assert "error" not in meta
    assert body.count('"token"') == 3, "each token must still stream incrementally"

    assert _threads_named_alive() <= before + 1, "worker should have exited on its own"


def test_provider_calls_are_bounded_by_a_client_timeout():
    """Cooperative cancellation cannot reach a provider that hangs BEFORE any token (
    on_token is never called). The client-side request timeout is the backstop; the SDK
    default is 600s, which would pin a thread for ten minutes."""
    import qaEngine

    assert qaEngine.OPENAI_TIMEOUT_SECONDS > 0
    assert qaEngine.OPENAI_TIMEOUT_SECONDS <= backend_main.STREAM_TOTAL_TIMEOUT, (
        "the provider timeout must sit under the stream deadline; past it nobody is "
        "listening to the response anyway"
    )
