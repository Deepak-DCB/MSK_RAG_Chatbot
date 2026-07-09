"""Model selection: /models catalog, config validation, and pinned routing.

Covers the UI-driven generation model picker:
  * `generation_catalog()` exposes free providers only when their server key is set,
    and always lists OpenAI as premium (requires the user's own key).
  * request config validates + threads `provider`/`model`.
  * `generate_answer_with_fallback` *pins* to the chosen provider+model and degrades
    straight to evidence-only on failure (never a silently different model).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "VectorDB"))

import qaEngine  # noqa: E402
import backend.main as main  # noqa: E402


def _clear_provider_env(monkeypatch):
    for definition in qaEngine._PROVIDER_DEFS.values():
        monkeypatch.delenv(definition["api_key_env"], raising=False)
        monkeypatch.delenv(definition["model_env"], raising=False)
    monkeypatch.delenv("FALLBACK_PROVIDERS", raising=False)


# ── generation_catalog ────────────────────────────────────────────────────────

def test_catalog_lists_openai_as_premium(monkeypatch):
    _clear_provider_env(monkeypatch)
    cat = qaEngine.generation_catalog()
    openai = next(p for p in cat["providers"] if p["name"] == "openai")
    assert openai["tier"] == "premium"
    assert openai["requires_user_key"] is True
    assert openai["models"]  # curated suggestions present


def test_catalog_exposes_free_provider_only_when_key_present(monkeypatch):
    _clear_provider_env(monkeypatch)
    cat = qaEngine.generation_catalog()
    groq = next(p for p in cat["providers"] if p["name"] == "groq")
    assert groq["tier"] == "free"
    assert groq["server_key"] is False
    assert cat["default_provider"] == "openai"  # no free provider configured

    monkeypatch.setenv("GROQ_API_KEY", "gk-" + "a" * 30)
    cat2 = qaEngine.generation_catalog()
    groq2 = next(p for p in cat2["providers"] if p["name"] == "groq")
    assert groq2["server_key"] is True
    assert cat2["default_provider"] == "groq"  # first configured free provider


# ── request config validation ─────────────────────────────────────────────────

@pytest.mark.parametrize("good", ["groq", "GROQ", "openai", "gemini"])
def test_sanitize_provider_accepts_known(good):
    assert main._sanitize_provider(good) == good.strip().lower()


@pytest.mark.parametrize("bad", [None, 123, "", "not-a-provider", "openai; drop"])
def test_sanitize_provider_rejects_unknown(bad):
    assert main._sanitize_provider(bad) is None


@pytest.mark.parametrize("good", [
    "gpt-4.1-mini", "llama-3.3-70b-versatile", "openai/gpt-oss-20b",
    "gemini-2.0-flash", "mistral-small-latest",
])
def test_sanitize_model_accepts_real_ids(good):
    assert main._sanitize_model(good) == good


@pytest.mark.parametrize("bad", [None, 123, "", "has space", "x" * 200, "bad\nnewline"])
def test_sanitize_model_rejects_malformed(bad):
    assert main._sanitize_model(bad) is None


def test_build_config_threads_provider_and_model():
    cfg, meta = main._build_config({"provider": "groq", "model": "llama-3.1-8b-instant"})
    assert cfg.generation_provider == "groq"
    assert cfg.generation_model == "llama-3.1-8b-instant"
    assert meta["generation_provider"] == "groq"
    assert meta["generation_model"] == "llama-3.1-8b-instant"
    assert meta["config_source"] == "request_override"


def test_build_config_drops_invalid_provider_and_model():
    cfg, _ = main._build_config({"provider": "nope", "model": "has space"})
    assert cfg.generation_provider is None
    assert cfg.generation_model is None


# ── pinned routing in generate_answer_with_fallback ───────────────────────────

class _Cfg:
    openai_model = "gpt-5.4-mini"
    num_predict = 100
    generation_provider = None
    generation_model = None


def _pin(provider, model):
    c = _Cfg()
    c.generation_provider = provider
    c.generation_model = model
    return c


def test_pinned_openai_uses_selected_model(monkeypatch):
    seen = {}

    def _fake_ask(prompt, model, num_predict, on_token=None, history=None, **k):
        seen["model"] = model
        if on_token:
            on_token("hi")
        return "hi", 1, 1

    monkeypatch.setattr(qaEngine, "ask_openai_llm", _fake_ask)
    text, pt, ot, mode, model = qaEngine.generate_answer_with_fallback(
        "p", _pin("openai", "gpt-4.1-nano"), context=[], context_pack=None, question="q",
    )
    assert mode == "llm:openai"
    assert model == "gpt-4.1-nano"
    assert seen["model"] == "gpt-4.1-nano"


def test_pinned_free_provider_routes_directly_without_touching_openai(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gk-" + "a" * 30)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    called = {}

    def _fake_call(spec, prompt, num_predict, on_token=None, history=None, system_prompt=None):
        called["name"] = spec.name
        called["model"] = spec.model
        if on_token:
            on_token("groq answer")
        return "groq answer", 2, 3

    monkeypatch.setattr(qaEngine, "_call_provider", _fake_call)

    def _openai_boom(*a, **k):
        raise AssertionError("OpenAI must not be called when a free provider is pinned")

    monkeypatch.setattr(qaEngine, "ask_openai_llm", _openai_boom)

    text, pt, ot, mode, model = qaEngine.generate_answer_with_fallback(
        "p", _pin("groq", "llama-3.1-8b-instant"), context=[], context_pack=None, question="q",
    )
    assert mode == "llm:groq"
    assert model == "llama-3.1-8b-instant"
    assert called == {"name": "groq", "model": "llama-3.1-8b-instant"}


def test_pinned_provider_without_server_key_degrades_to_evidence_only(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    text, pt, ot, mode, model = qaEngine.generate_answer_with_fallback(
        "p", _pin("groq", "whatever"), context=[], context_pack=None, question="q",
    )
    assert mode == "evidence_only"
    assert model is None


def test_pinned_openai_dead_key_degrades_to_evidence_only(monkeypatch):
    def _dead(*a, **k):
        raise qaEngine.OpenAIKeyError("dead")

    monkeypatch.setattr(qaEngine, "ask_openai_llm", _dead)
    text, pt, ot, mode, model = qaEngine.generate_answer_with_fallback(
        "p", _pin("openai", "gpt-4.1-mini"), context=[], context_pack=None, question="q",
    )
    assert mode == "evidence_only"
    assert model is None


# ── /models endpoint ──────────────────────────────────────────────────────────

def test_models_endpoint_returns_catalog():
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data.get("providers"), list) and data["providers"]
    assert any(p["name"] == "openai" for p in data["providers"])
    assert "default_provider" in data and "default_model" in data
