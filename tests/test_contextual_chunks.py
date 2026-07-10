"""Zero-cost tests for the contextual-retrieval augmentation pipeline.

These exercise only the pure, deterministic parts (prompt construction, token
budgeting, and the store/collection env override). No API calls, no artifacts.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts", PROJECT_ROOT / "VectorDB"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_contextual_chunks as bcc


def test_context_prefix_prompt_names_article_and_section():
    prompt = bcc.build_context_prefix_prompt(
        title="How to truly treat thoracic outlet syndrome",
        section="Scalene function",
        article_summary="Covers TOS mechanisms and scapular correction.",
        chunk_text="This muscle is usually inhibited rather than tight.",
    )
    assert "thoracic outlet syndrome" in prompt
    assert "Scalene function" in prompt
    assert "Covers TOS mechanisms" in prompt
    # It must instruct a short, grounded, single sentence.
    assert "ONE short sentence" in prompt
    assert "Do NOT add facts not present" in prompt


def test_context_prefix_prompt_handles_missing_section():
    prompt = bcc.build_context_prefix_prompt("T", "", "S", "chunk")
    assert "(none)" in prompt


def test_article_summary_prompt_truncates_long_article():
    long_text = "word " * 50_000  # far over the token budget
    prompt = bcc.build_article_summary_prompt("Title", long_text)
    # The whole thing must be bounded well under the raw article length.
    assert bcc.count_tokens(prompt) <= bcc.ARTICLE_TEXT_BUDGET_TOKENS + 400


def test_truncate_tokens_is_bounded():
    assert bcc.count_tokens(bcc.truncate_tokens("a b c d e f g", 3)) <= 3


def test_qaengine_store_env_override(monkeypatch):
    """Setting MSK_CHROMA_DIR / MSK_COLLECTION redirects the store; unset = prod."""
    monkeypatch.setenv("MSK_CHROMA_DIR", r"/tmp/some_contextual_store")
    monkeypatch.setenv("MSK_COLLECTION", "msk_chunks_contextual")
    sys.modules.pop("qaEngine", None)
    qa = importlib.import_module("qaEngine")
    try:
        assert qa.COLLECTION_NAME == "msk_chunks_contextual"
        assert qa.PERSIST_DIR.endswith("some_contextual_store")
    finally:
        # Reset to production defaults for any later importer in the session.
        monkeypatch.delenv("MSK_CHROMA_DIR", raising=False)
        monkeypatch.delenv("MSK_COLLECTION", raising=False)
        sys.modules.pop("qaEngine", None)
        prod = importlib.import_module("qaEngine")
        assert prod.COLLECTION_NAME == "msk_chunks"
