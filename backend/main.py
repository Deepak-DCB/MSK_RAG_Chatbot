#!/usr/bin/env python3
"""
backend/main.py — FastAPI backend for MSK Triage Chatbot
Deployed on Render Free tier.
"""

import os
import sys
import time
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Path setup so VectorDB/ is importable ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "VectorDB"))

from qaEngine import agentic_run, _backend, QAConfig  # noqa: E402

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MSK Triage Chatbot API",
    description="RAG-powered musculoskeletal neurology Q&A backend",
    version="1.0.0",
)

# ── CORS — allow Vercel frontend ──────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5500,http://127.0.0.1:5500"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Safety constants ──────────────────────────────────────────────────────────
MAX_QUESTION_LEN = 1000
MAX_HISTORY_TURNS = 5
MAX_OUTPUT_TOKENS = 1000
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 5      # requests per window per IP

# ── In-memory rate limiter ────────────────────────────────────────────────────
_rate_log: Dict[str, collections.deque] = {}


def _check_rate_limit(ip: str) -> None:
    now = time.time()
    if ip not in _rate_log:
        _rate_log[ip] = collections.deque()
    dq = _rate_log[ip]
    # Purge old entries
    while dq and dq[0] < now - RATE_LIMIT_WINDOW:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW}s.",
        )
    dq.append(now)


# ── Startup: preload Chroma ──────────────────────────────────────────────────
@app.on_event("startup")
def startup_load():
    _backend.load_collection()


# ── Schemas ───────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_LEN)
    history: Optional[List[Dict[str, str]]] = Field(default=None)
    config: Optional[Dict[str, Any]] = Field(default=None)


class AskResponse(BaseModel):
    answer: str
    citations: List[str]
    retrieval_confidence: float
    retrieval_time: float
    generation_time: float
    prompt_tokens: int
    output_tokens: int
    context_tokens: int
    question_tokens: int
    category: Optional[str] = None
    category_label: Optional[str] = None
    refined_query: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    coll = _backend.collection
    return {
        "status": "ok",
        "chroma_loaded": coll is not None,
        "chunk_count": coll.count() if coll else 0,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    # Rate limit
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    # Validate question
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Trim history
    history = None
    if req.history:
        history = req.history[-MAX_HISTORY_TURNS:]

    # Build config with safety caps
    cfg_dict = req.config or {}
    cfg_dict["num_predict"] = min(cfg_dict.get("num_predict", MAX_OUTPUT_TOKENS), MAX_OUTPUT_TOKENS)
    cfg_dict.setdefault("use_reranker", False)  # save LLM cost by default

    try:
        cfg = QAConfig(**{k: v for k, v in cfg_dict.items() if hasattr(QAConfig, k)})
    except Exception:
        cfg = QAConfig(num_predict=MAX_OUTPUT_TOKENS, use_reranker=False)

    # Run RAG
    try:
        res = agentic_run(question, cfg=cfg, history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine error: {e}")

    return AskResponse(
        answer=res.get("answer", ""),
        citations=res.get("citations", []),
        retrieval_confidence=res.get("retrieval_confidence", 0.0),
        retrieval_time=res.get("retrieval_time", 0.0),
        generation_time=res.get("generation_time", 0.0),
        prompt_tokens=res.get("prompt_tokens", 0),
        output_tokens=res.get("output_tokens", 0),
        context_tokens=res.get("context_tokens", 0),
        question_tokens=res.get("question_tokens", 0),
        category=res.get("category"),
        category_label=res.get("category_label"),
        refined_query=res.get("refined_query"),
    )
