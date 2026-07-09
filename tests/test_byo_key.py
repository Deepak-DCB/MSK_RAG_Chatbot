"""Phase 7: bring-your-own-key handling.

Verifies the key is (a) validated robustly (not just an sk- prefix), (b) threaded to
the OpenAI client for the request only, (c) reset afterwards, and (d) never leaked into
telemetry / config metadata.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "VectorDB"))

import backend.main as main  # noqa: E402
import qaEngine  # noqa: E402


# ── Validation ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("good", [
    "sk-" + "a" * 40,
    "sk-proj-" + "A1b2C3d4" * 6,
    "abc123DEF456ghi789JKL012",  # not sk- prefixed, still structurally a key
])
def test_sanitize_accepts_structurally_valid_keys(good):
    assert main._sanitize_user_api_key(good) == good


@pytest.mark.parametrize("bad", [
    None,
    123,
    "",
    "short",
    "has spaces in it aaaaaaaaaaaaaaaaaaaa",
    "sk-with\nnewline-aaaaaaaaaaaaaaaaaaaa",
    "quotes\"aaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "x" * 600,  # overlong
])
def test_sanitize_rejects_malformed_keys(bad):
    assert main._sanitize_user_api_key(bad) is None


# ── _build_config ─────────────────────────────────────────────────────────────

def test_build_config_accepts_valid_key_but_hides_it_from_meta():
    key = "sk-" + "z" * 40
    cfg, meta = main._build_config({"api_key": key})
    assert cfg.api_key == key
    assert meta["user_key_active"] is True
    assert meta["config_source"] == "user_key"
    # The key value must not appear anywhere in the telemetry meta.
    assert key not in str(meta)
    for v in meta.values():
        assert key != v


def test_build_config_drops_malformed_key():
    cfg, meta = main._build_config({"api_key": "nope"})
    assert cfg.api_key is None
    assert meta["user_key_active"] is False


def test_build_config_key_not_in_config_repr():
    key = "sk-" + "q" * 40
    cfg, _ = main._build_config({"api_key": key})
    assert key not in repr(cfg)  # api_key field uses repr=False


# ── ContextVar threading in qaEngine ──────────────────────────────────────────

def test_request_key_builds_ephemeral_client_and_resets(monkeypatch):
    captured = {}

    def _fake_ctor(**kwargs):
        captured["api_key"] = kwargs.get("api_key")
        return object()

    monkeypatch.setattr(qaEngine, "OpenAI", _fake_ctor)
    monkeypatch.setattr(qaEngine, "_openai_client", None, raising=False)

    token = qaEngine._request_api_key.set("sk-user-key-abc")
    try:
        qaEngine._get_openai_client()
        assert captured["api_key"] == "sk-user-key-abc"  # ephemeral client used the user key
    finally:
        qaEngine._request_api_key.reset(token)

    # After reset, no per-request key is bound.
    assert qaEngine._request_api_key.get() is None


def test_agentic_run_binds_and_unbinds_key(monkeypatch):
    seen = {}

    def _fake_impl(question, cfg=None, history=None, on_token=None):
        seen["during"] = qaEngine._request_api_key.get()
        return {"answer": "ok"}

    monkeypatch.setattr(qaEngine, "_agentic_run_impl", _fake_impl)
    cfg = qaEngine.QAConfig(api_key="sk-abc-key-123")
    qaEngine.agentic_run("q", cfg=cfg)

    assert seen["during"] == "sk-abc-key-123"     # bound during the request
    assert qaEngine._request_api_key.get() is None  # unbound afterwards
