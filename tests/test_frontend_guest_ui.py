from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frontend_remains_guest_only_and_privacy_copy_is_visible():
    app_js = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    # The chat UI (and its privacy copy) lives in chat.html; index.html is the landing page.
    chat_html = (PROJECT_ROOT / "frontend" / "chat.html").read_text(encoding="utf-8")

    assert "const AUTH_ENABLED = false" in app_js
    assert "Guest mode: chats are not saved by this app" in chat_html
    assert "Sign in" not in chat_html
    assert "save history" not in chat_html.lower()


def test_frontend_trust_ui_controls_are_present():
    app_js = (PROJECT_ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    index_html = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

    assert "Why this answer?" in app_js
    assert "Sources (" in app_js
    assert "Felt unsafe" in app_js
    assert "Not grounded" in app_js
    assert "Local feedback selected:" in app_js
    assert "Mechanics Study" in index_html
    assert "/study/mechanics" in app_js
    assert "Mechanics map" in app_js
