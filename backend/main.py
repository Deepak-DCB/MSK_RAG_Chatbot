#!/usr/bin/env python3
"""
backend/main.py — FastAPI backend for MSK Triage Chatbot
Deployed on Render Free tier.
"""

import os
import sys
import time
import json
import queue
import threading
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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
ALLOWED_ORIGINS = [
    "https://msk-rag-chatbot.vercel.app",
    "https://msk-triage-chatbot.vercel.app",
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

extra = os.getenv("CORS_ORIGINS", "")
if extra:
    ALLOWED_ORIGINS.extend([o.strip() for o in extra.split(",") if o.strip()])

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
    while dq and dq[0] < now - RATE_LIMIT_WINDOW:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW}s.",
        )
    dq.append(now)


# ── Supabase setup ────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

_supabase_client = None


def _get_supabase():
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None
    try:
        from supabase import create_client
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        return _supabase_client
    except Exception:
        return None


def _extract_user_id(authorization: Optional[str]) -> Optional[str]:
    """Extract user_id from Supabase JWT. Returns None if not authenticated."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    try:
        import jwt
        # Decode using the JWT secret from Supabase
        # The JWT secret is in Settings → API → JWT Secret
        secret = SUPABASE_JWT_SECRET
        if not secret:
            # Fallback: extract from anon key (not ideal but works for verification)
            # For proper setup, set SUPABASE_JWT_SECRET env var
            return None
        payload = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
        return payload.get("sub")
    except Exception:
        return None


def _save_conversation(user_id: str, question: str, answer: str,
                       citations: List[str], category: Optional[str],
                       confidence: float):
    """Save a conversation to Supabase (fire-and-forget)."""
    sb = _get_supabase()
    if not sb or not user_id:
        return
    try:
        sb.table("conversations").insert({
            "user_id": user_id,
            "question": question,
            "answer": answer,
            "citations": citations,
            "category": category or "",
            "confidence": confidence,
        }).execute()
    except Exception:
        pass  # Don't break the response if DB write fails


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


# ── Helper: build config with safety caps ─────────────────────────────────────
def _build_config(cfg_dict: Optional[Dict[str, Any]]) -> QAConfig:
    cfg_dict = cfg_dict or {}
    cfg_dict["num_predict"] = min(cfg_dict.get("num_predict", MAX_OUTPUT_TOKENS), MAX_OUTPUT_TOKENS)
    cfg_dict.setdefault("use_reranker", False)
    try:
        return QAConfig(**{k: v for k, v in cfg_dict.items() if hasattr(QAConfig, k)})
    except Exception:
        return QAConfig(num_predict=MAX_OUTPUT_TOKENS, use_reranker=False)


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
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = req.history[-MAX_HISTORY_TURNS:] if req.history else None
    cfg = _build_config(req.config)

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


# ── Streaming endpoint (SSE) ─────────────────────────────────────────────────
_SENTINEL = object()


@app.post("/ask/stream")
def ask_stream(req: AskRequest, request: Request,
               authorization: Optional[str] = Header(None)):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Extract user_id from JWT (optional — guests can still use)
    user_id = _extract_user_id(authorization)

    history = req.history[-MAX_HISTORY_TURNS:] if req.history else None
    cfg = _build_config(req.config)

    token_q: queue.Queue = queue.Queue()
    result_holder: Dict[str, Any] = {}

    def on_token(tok: str):
        token_q.put(tok)

    def run_engine():
        try:
            res = agentic_run(question, cfg=cfg, history=history, on_token=on_token)
            result_holder.update(res)
        except Exception as e:
            result_holder["error"] = str(e)
        finally:
            token_q.put(_SENTINEL)

    threading.Thread(target=run_engine, daemon=True).start()

    def event_stream():
        while True:
            try:
                item = token_q.get(timeout=120)
            except queue.Empty:
                yield "event: done\ndata: {\"error\": \"Timeout\"}\n\n"
                return
            if item is _SENTINEL:
                break
            yield f"data: {json.dumps({'token': item})}\n\n"

        # Send final metadata
        meta = {
            "citations": result_holder.get("citations", []),
            "retrieval_confidence": result_holder.get("retrieval_confidence", 0.0),
            "retrieval_time": result_holder.get("retrieval_time", 0.0),
            "generation_time": result_holder.get("generation_time", 0.0),
            "prompt_tokens": result_holder.get("prompt_tokens", 0),
            "output_tokens": result_holder.get("output_tokens", 0),
            "context_tokens": result_holder.get("context_tokens", 0),
            "question_tokens": result_holder.get("question_tokens", 0),
            "category": result_holder.get("category"),
            "category_label": result_holder.get("category_label"),
            "refined_query": result_holder.get("refined_query"),
        }
        if "error" in result_holder:
            meta["error"] = result_holder["error"]
        yield f"event: done\ndata: {json.dumps(meta)}\n\n"

        # Save to Supabase (after stream completes)
        if user_id and "answer" in result_holder:
            threading.Thread(
                target=_save_conversation,
                args=(
                    user_id,
                    question,
                    result_holder.get("answer", ""),
                    result_holder.get("citations", []),
                    result_holder.get("category"),
                    result_holder.get("retrieval_confidence", 0.0),
                ),
                daemon=True,
            ).start()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── History endpoint ──────────────────────────────────────────────────────────
@app.get("/history")
def get_history(authorization: Optional[str] = Header(None),
                limit: int = 50, offset: int = 0):
    user_id = _extract_user_id(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    sb = _get_supabase()
    if not sb:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        resp = (
            sb.table("conversations")
            .select("id, question, answer, citations, category, confidence, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return {"conversations": resp.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
