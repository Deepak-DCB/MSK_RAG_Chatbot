"""Zero-cost tests for the pluggable embedding backend.

Covers the choke point that lets a fully non-OpenAI pipeline run: the default is
OpenAI (production untouched), MSK_EMBED_PROVIDER=local routes to a local
SentenceTransformer, and every query embedding flows through one dispatcher so a
non-OpenAI pipeline can never silently call OpenAI. No network, no model download.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts", PROJECT_ROOT / "VectorDB"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import qaEngine  # noqa: E402


class _FakeSTModel:
    """Stands in for a SentenceTransformer — returns deterministic unit vectors."""
    def __init__(self, dim: int = 4):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **kw):
        # One already-normalized vector per input (first axis hot).
        out = []
        for i, _ in enumerate(texts):
            v = [0.0] * self.dim
            v[i % self.dim] = 1.0
            out.append(v)
        return out


# ── default is OpenAI (production untouched) ──────────────────────────────────

def test_default_provider_is_openai_when_env_unset(monkeypatch):
    monkeypatch.delenv("MSK_EMBED_PROVIDER", raising=False)
    sys.modules.pop("qaEngine", None)
    qa = importlib.import_module("qaEngine")
    try:
        assert qa.EMBED_PROVIDER == "openai"
        assert qa.EMBED_MODEL == "text-embedding-3-large"
    finally:
        sys.modules.pop("qaEngine", None)
        importlib.import_module("qaEngine")


def test_embed_texts_routes_to_openai_by_default(monkeypatch):
    calls = {}

    def _fake_openai(texts):
        calls["texts"] = texts
        return [[0.1]]

    monkeypatch.setattr(qaEngine, "EMBED_PROVIDER", "openai")
    monkeypatch.setattr(qaEngine, "openai_embed", _fake_openai)

    def _local_boom(_):
        raise AssertionError("local backend must not be touched when provider=openai")

    monkeypatch.setattr(qaEngine, "local_embed", _local_boom)
    out = qaEngine.embed_texts(["hello"])
    assert out == [[0.1]]
    assert calls["texts"] == ["hello"]


# ── local backend selected ────────────────────────────────────────────────────

def test_embed_texts_routes_to_local_when_selected(monkeypatch):
    monkeypatch.setattr(qaEngine, "EMBED_PROVIDER", "local")
    monkeypatch.setattr(qaEngine, "_get_local_embedder", lambda: _FakeSTModel(dim=4))

    def _openai_boom(_):
        raise AssertionError("OpenAI must not be called when provider=local")

    monkeypatch.setattr(qaEngine, "openai_embed", _openai_boom)
    out = qaEngine.embed_texts(["a", "b"])
    assert len(out) == 2 and len(out[0]) == 4
    assert all(isinstance(x, float) for x in out[0])  # coerced to float
    assert out[0] == [1.0, 0.0, 0.0, 0.0]


def test_encode_query_flows_through_backend(monkeypatch):
    monkeypatch.setattr(qaEngine, "EMBED_PROVIDER", "local")
    monkeypatch.setattr(qaEngine, "_get_local_embedder", lambda: _FakeSTModel(dim=3))
    monkeypatch.setattr(qaEngine, "openai_embed",
                        lambda _: (_ for _ in ()).throw(AssertionError("no OpenAI")))
    nested = qaEngine.encode_query("why does my shoulder ache?")
    assert nested == [[1.0, 0.0, 0.0]]  # single query -> one nested vector


def test_local_embed_handles_blank_text(monkeypatch):
    monkeypatch.setattr(qaEngine, "_get_local_embedder", lambda: _FakeSTModel(dim=2))
    out = qaEngine.local_embed(["", "   "])
    assert len(out) == 2 and all(len(v) == 2 for v in out)


# ── build script: corpus filter matches production ────────────────────────────

def test_build_local_store_is_bad_filter():
    import build_local_embed_store as b
    assert b.is_bad("") is True
    assert b.is_bad("short") is True                      # under MIN_LEN
    assert b.is_bad("See Figure 3 for details") is True   # caption-like
    good = "The scalene muscles form the interscalene triangle through which the plexus passes."
    assert b.is_bad(good) is False
