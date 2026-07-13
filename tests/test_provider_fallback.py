"""Phase 6: free-provider fallback chain + per-provider adapters.

These exercise provider selection, the OpenAI-compatible adapter (streaming vs
non-streaming, param style), correct client construction per provider, and the
OpenAI -> providers -> evidence-only orchestration including the mid-stream guard.
No live provider calls are made; a fake OpenAI client is injected.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "VectorDB"))

import qaEngine  # noqa: E402


# ── Fakes ────────────────────────────────────────────────────────────────────

class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, streaming):
        if streaming:
            self.delta = _Delta(content)
        else:
            self.message = type("M", (), {"content": content})()


class _Chunk:
    def __init__(self, content):
        self.choices = [_Choice(content, streaming=True)]


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content, streaming=False)]


class _FakeCompletions:
    def __init__(self, parent):
        self.parent = parent

    def create(self, **kwargs):
        self.parent.calls.append(kwargs)
        if kwargs.get("stream"):
            if self.parent.stream_error:
                raise self.parent.stream_error
            return iter([_Chunk(t) for t in self.parent.tokens])
        if self.parent.nonstream_error:
            raise self.parent.nonstream_error
        return _Resp("".join(self.parent.tokens))


class _FakeClient:
    def __init__(self, tokens=("hello", " world"), stream_error=None, nonstream_error=None):
        self.tokens = list(tokens)
        self.stream_error = stream_error
        self.nonstream_error = nonstream_error
        self.calls = []
        self.chat = type("C", (), {"completions": _FakeCompletions(self)})()


# ── Provider registry ────────────────────────────────────────────────────────

def _clear_provider_env(monkeypatch):
    for name in qaEngine._PROVIDER_DEFS.values():
        monkeypatch.delenv(name["api_key_env"], raising=False)
        monkeypatch.delenv(name["model_env"], raising=False)
    monkeypatch.delenv("FALLBACK_PROVIDERS", raising=False)


def test_configured_providers_only_includes_those_with_keys(monkeypatch):
    _clear_provider_env(monkeypatch)
    assert qaEngine._configured_providers() == []

    monkeypatch.setenv("GROQ_API_KEY", "gk")
    specs = qaEngine._configured_providers()
    assert [s.name for s in specs] == ["groq"]
    assert specs[0].base_url == "https://api.groq.com/openai/v1"
    assert specs[0].model  # default model applied
    assert specs[0].api_key == "gk"


def test_configured_providers_respects_explicit_order_and_model_override(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gk")
    monkeypatch.setenv("CEREBRAS_API_KEY", "ck")
    monkeypatch.setenv("CEREBRAS_MODEL", "my-cerebras-model")
    monkeypatch.setenv("FALLBACK_PROVIDERS", "cerebras,groq")

    specs = qaEngine._configured_providers()
    assert [s.name for s in specs] == ["cerebras", "groq"]
    assert specs[0].model == "my-cerebras-model"


# ── Adapter: ask_openai_llm with injected client ─────────────────────────────

def test_ask_openai_llm_streaming_with_injected_client_emits_tokens():
    fake = _FakeClient(tokens=["a", "b", "c"])
    emitted = []
    answer, pt, ot = qaEngine.ask_openai_llm(
        "prompt", model="some-model", num_predict=50,
        on_token=emitted.append, client=fake, param_style="compat",
    )
    assert answer == "abc"
    assert emitted == ["a", "b", "c"]
    # compat style must use max_tokens, not max_completion_tokens
    assert "max_tokens" in fake.calls[0]
    assert "max_completion_tokens" not in fake.calls[0]


def test_ask_openai_llm_non_streaming_provider_emits_once():
    fake = _FakeClient(tokens=["done"])
    emitted = []
    answer, pt, ot = qaEngine.ask_openai_llm(
        "prompt", model="m", num_predict=50,
        on_token=emitted.append, client=fake, supports_streaming=False, param_style="compat",
    )
    assert answer == "done"
    assert emitted == ["done"]  # emitted exactly once via the non-streaming path
    assert all(not c.get("stream") for c in fake.calls)  # never attempted streaming


def test_cancelled_stream_does_not_retry_after_a_pre_token_timeout():
    class Cancelled(BaseException):
        pass

    fake = _FakeClient(tokens=["never delivered"], stream_error=TimeoutError("provider timeout"))

    def check_cancelled():
        if fake.calls:
            raise Cancelled()

    def on_token(_token):
        raise AssertionError("no token should be delivered")

    on_token.check_cancelled = check_cancelled

    with pytest.raises(Cancelled):
        qaEngine.ask_openai_llm(
            "prompt",
            model="some-model",
            num_predict=50,
            on_token=on_token,
            client=fake,
            param_style="compat",
        )

    assert len(fake.calls) == 1
    assert fake.calls[0]["stream"] is True


def test_ask_openai_llm_openai_style_uses_max_completion_tokens():
    fake = _FakeClient(tokens=["x"])
    qaEngine.ask_openai_llm("p", model="gpt-4.1-mini", num_predict=10, client=fake, param_style="openai")
    assert "max_completion_tokens" in fake.calls[0]


# ── _call_provider builds the right client ───────────────────────────────────

def test_call_provider_constructs_client_with_base_url_and_headers(monkeypatch):
    captured = {}

    def _fake_openai_ctor(**kwargs):
        captured.update(kwargs)
        return _FakeClient(tokens=["ok"])

    monkeypatch.setattr(qaEngine, "OpenAI", _fake_openai_ctor)
    spec = qaEngine.ProviderSpec(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="or-key",
        model="openrouter/auto",
        extra_headers={"X-Title": "MSK Triage Chatbot"},
    )
    answer, _, _ = qaEngine._call_provider(spec, "prompt", 50)
    assert answer == "ok"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["api_key"] == "or-key"
    assert captured["default_headers"] == {"X-Title": "MSK Triage Chatbot"}


# ── Orchestration: generate_answer_with_fallback ─────────────────────────────

class _Cfg:
    openai_model = "gpt-4.1-mini"
    num_predict = 100
    generation_provider = None
    generation_model = None


def test_fallback_uses_first_working_provider(monkeypatch):
    # OpenAI primary is dead.
    def _openai_dead(*a, **k):
        raise qaEngine.OpenAIKeyError("dead")
    monkeypatch.setattr(qaEngine, "ask_openai_llm", _openai_dead)

    specs = [
        qaEngine.ProviderSpec("groq", "u", "k", "m"),
        qaEngine.ProviderSpec("cerebras", "u", "k", "m"),
    ]
    monkeypatch.setattr(qaEngine, "_configured_providers", lambda: specs)

    def _fake_call(spec, prompt, num_predict, on_token=None, history=None, system_prompt=None):
        if spec.name == "groq":
            raise RuntimeError("groq down")
        if on_token:
            on_token("via cerebras")
        return "via cerebras", 1, 2
    monkeypatch.setattr(qaEngine, "_call_provider", _fake_call)

    got = []
    text, pt, ot, mode, model = qaEngine.generate_answer_with_fallback(
        "prompt", _Cfg(), context=[], context_pack=None, question="q", on_token=got.append,
    )
    assert mode == "llm:cerebras"
    assert text == "via cerebras"
    assert model == "m"          # the cerebras spec's model is reported
    assert got == ["via cerebras"]


def test_fallback_to_evidence_only_when_all_providers_fail(monkeypatch):
    monkeypatch.setattr(qaEngine, "ask_openai_llm", lambda *a, **k: (_ for _ in ()).throw(qaEngine.OpenAIKeyError("dead")))
    monkeypatch.setattr(qaEngine, "_configured_providers", lambda: [])

    text, pt, ot, mode, model = qaEngine.generate_answer_with_fallback(
        "prompt", _Cfg(), context=[], context_pack=None, question="q",
    )
    assert mode == "evidence_only"
    assert model is None
    assert "in-person" in text.lower()


def test_empty_openai_answer_degrades_to_evidence_only(monkeypatch):
    # OpenAI returns HTTP 200 with empty content (e.g. a reasoning model that
    # spent its whole budget) and streams zero tokens. This must NOT surface as a
    # dead "No response received" bubble — it degrades to the evidence-only answer.
    def _empty_ok(prompt, model, num_predict, on_token=None, history=None, **k):
        return "", 5, 0  # success, but no content, no tokens emitted
    monkeypatch.setattr(qaEngine, "ask_openai_llm", _empty_ok)
    monkeypatch.setattr(qaEngine, "_configured_providers", lambda: [])

    got = []
    text, pt, ot, mode, model = qaEngine.generate_answer_with_fallback(
        "prompt", _Cfg(), context=[], context_pack=None, question="q", on_token=got.append,
    )
    assert mode == "evidence_only"
    assert model is None
    assert text.strip()          # never empty
    assert "".join(got).strip()  # something was streamed to the client
    assert "in-person" in text.lower()


def test_empty_provider_answer_falls_through_to_next(monkeypatch):
    # OpenAI empty -> first free provider empty -> second provider answers.
    monkeypatch.setattr(qaEngine, "ask_openai_llm", lambda *a, **k: ("", 5, 0))
    specs = [
        qaEngine.ProviderSpec("groq", "u", "k", "m"),
        qaEngine.ProviderSpec("cerebras", "u", "k", "m2"),
    ]
    monkeypatch.setattr(qaEngine, "_configured_providers", lambda: specs)

    def _fake_call(spec, prompt, num_predict, on_token=None, history=None, system_prompt=None):
        if spec.name == "groq":
            return "   ", 1, 0  # whitespace-only, zero tokens => empty
        if on_token:
            on_token("real answer")
        return "real answer", 1, 2
    monkeypatch.setattr(qaEngine, "_call_provider", _fake_call)

    text, pt, ot, mode, model = qaEngine.generate_answer_with_fallback(
        "prompt", _Cfg(), context=[], context_pack=None, question="q", on_token=lambda t: None,
    )
    assert mode == "llm:cerebras"
    assert text == "real answer"


class _PinnedGroqCfg:
    openai_model = "gpt-4.1-mini"
    num_predict = 100
    generation_provider = "groq"   # user pinned Groq in the model dropdown
    generation_model = "llama-3.3-70b-versatile"


def test_pinned_provider_empty_answer_degrades_to_evidence_only(monkeypatch):
    # Reproduces the reported bug: a pinned Groq model returns HTTP 200 with empty
    # content, streaming zero tokens. Before the fix this returned "" as a success
    # and the client showed a dead "No response received." bubble. The pinned
    # contract says: the chosen model, or a safe grounded answer — so it must land
    # on evidence-only, never an empty success.
    spec = qaEngine.ProviderSpec("groq", "https://api.groq.com/openai/v1", "gk",
                                 "llama-3.3-70b-versatile")
    monkeypatch.setattr(qaEngine, "_provider_spec_for", lambda name, model=None: spec)

    def _empty_call(spec, prompt, num_predict, on_token=None, history=None, system_prompt=None):
        return "", 5, 0  # 200 OK, no content, nothing streamed
    monkeypatch.setattr(qaEngine, "_call_provider", _empty_call)

    got = []
    text, pt, ot, mode, model = qaEngine.generate_answer_with_fallback(
        "prompt", _PinnedGroqCfg(), context=[], context_pack=None, question="q",
        on_token=got.append,
    )
    assert mode == "evidence_only"
    assert model is None
    assert text.strip()          # real, non-empty grounded answer
    assert "".join(got).strip()  # streamed to the client (no dead bubble)


def test_fallback_reraises_if_tokens_already_streamed(monkeypatch):
    # OpenAI emits a token then dies — we must NOT silently switch providers.
    def _partial_then_die(prompt, model, num_predict, on_token=None, history=None, **k):
        if on_token:
            on_token("partial")
        raise qaEngine.OpenAIKeyError("died mid-stream")
    monkeypatch.setattr(qaEngine, "ask_openai_llm", _partial_then_die)
    monkeypatch.setattr(qaEngine, "_configured_providers", lambda: [qaEngine.ProviderSpec("groq", "u", "k", "m")])

    with pytest.raises(qaEngine.OpenAIKeyError):
        qaEngine.generate_answer_with_fallback(
            "prompt", _Cfg(), context=[], context_pack=None, question="q", on_token=lambda t: None,
        )
