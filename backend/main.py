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
import ipaddress
import re
import uuid
import logging
import threading
import collections
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── Path setup so VectorDB/ is importable ─────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "VectorDB"))

from qaEngine import (  # noqa: E402
    OPENAI_MODEL,
    OpenAIKeyError,
    QAConfig,
    _backend,
    _red_flag_response,
    _scope_boundary_response,
    agentic_run,
    detect_red_flags,
    detect_scope_issue,
    generation_catalog,
)
from mechanics_retrieval import build_mechanics_context  # noqa: E402

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MSK Triage Chatbot API",
    description="RAG-powered musculoskeletal neurology Q&A backend",
    version="1.0.0",
)

logger = logging.getLogger(__name__)

# ── CORS — allow Vercel frontend ──────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "https://msk-rag-chatbot.vercel.app",
    "https://msk-triage-chatbot.vercel.app",
    "https://mskchat.vercel.app",
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
MAX_SUMMARY_LEN = 2500  # rolling conversation summary (client-supplied, truncated)
# Lower budget = more faithful answers. The model bake-off (scripts/model_bakeoff.py)
# showed gpt-oss faithfulness rises sharply as the budget shrinks (0.25 @2048 ->
# 0.64 @700): a larger budget lets the model ramble into confident, unsupported
# mechanisms. gpt-oss answers finish at <=380 tokens under a 700 cap, so this curbs
# rambling without truncating real answers.
MAX_OUTPUT_TOKENS = 700
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 5      # requests per window per IP
MECHANICS_MAX_ITEMS_DEFAULT = 8
MECHANICS_MAX_ITEMS_MIN = 1
MECHANICS_MAX_ITEMS_MAX = 12
PUBLIC_RERANKER_TOP_N_MIN = 1
PUBLIC_RERANKER_TOP_N_MAX = 10
MAX_HISTORY_OFFSET = 5000
MAX_HISTORY_MESSAGES = MAX_HISTORY_TURNS * 2
MAX_HISTORY_MSG_LEN = 4000  # per-message cap; qaEngine truncates further before prompting
RATE_LOG_PRUNE_THRESHOLD = 1024  # prune drained IP buckets once the map grows past this
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "").strip().lower() in {"1", "true", "yes", "on"}
TRUSTED_PROXY_NETWORKS = []
for raw_proxy in os.getenv("TRUSTED_PROXY_IPS", "").split(","):
    raw_proxy = raw_proxy.strip()
    if not raw_proxy:
        continue
    try:
        TRUSTED_PROXY_NETWORKS.append(ipaddress.ip_network(raw_proxy, strict=False))
    except ValueError:
        logger.warning("Ignoring invalid TRUSTED_PROXY_IPS entry: %s", raw_proxy)

# ── In-memory rate limiter ────────────────────────────────────────────────────
_rate_log: Dict[str, collections.deque] = {}
_rate_log_lock = threading.Lock()


def _prune_rate_log(now: float) -> None:
    """Drop IPs whose window has fully drained. Without this, _rate_log keeps one entry
    per unique IP forever — an IPv6 /64 gives an attacker 2^64 of them."""
    cutoff = now - RATE_LIMIT_WINDOW
    stale = [ip for ip, dq in _rate_log.items() if not dq or dq[-1] < cutoff]
    for ip in stale:
        del _rate_log[ip]


def _check_rate_limit(ip: str) -> None:
    now = time.time()
    with _rate_log_lock:
        if len(_rate_log) > RATE_LOG_PRUNE_THRESHOLD:
            _prune_rate_log(now)
        if ip not in _rate_log:
            _rate_log[ip] = collections.deque()
        dq = _rate_log[ip]
        while dq and dq[0] < now - RATE_LIMIT_WINDOW:
            dq.popleft()
        if len(dq) >= RATE_LIMIT_MAX:
            request_id = _new_request_id()
            logger.warning("rate limit exceeded [request_id=%s ip=%s]", request_id, ip)
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded. Max {RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW}s. "
                    f"request_id={request_id}. If you have severe neurologic symptoms, chest pain, "
                    "or other red flags, seek urgent in-person care."
                ),
                headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
            )
        dq.append(now)


def _client_ip(request: Request) -> str:
    """Resolve the client IP for rate limiting.

    When TRUST_PROXY_HEADERS is on we read X-Forwarded-For **right to left**. A platform
    proxy (Render, Cloudflare) *appends* the peer it saw to whatever XFF the client sent,
    so the rightmost entry is the only one the client cannot forge — reading the leftmost
    would let anyone spoof an arbitrary IP and evade (or poison) the limiter.

    TRUSTED_PROXY_IPS optionally restricts *which* direct peers may be believed. Leaving
    it unset no longer disables the header entirely (previously `any()` over an empty list
    was always False, so TRUST_PROXY_HEADERS=1 silently did nothing and every user behind
    the proxy shared one rate-limit bucket); it now means "trust the immediate peer",
    which is the platform proxy in a standard single-hop deployment.
    """
    direct_ip = request.client.host if request.client else "unknown"
    if not TRUST_PROXY_HEADERS:
        return direct_ip
    try:
        direct_addr = ipaddress.ip_address(direct_ip)
    except ValueError:
        return direct_ip
    if TRUSTED_PROXY_NETWORKS and not any(direct_addr in network for network in TRUSTED_PROXY_NETWORKS):
        return direct_ip

    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        for candidate in reversed([p.strip() for p in forwarded_for.split(",")]):
            if not candidate:
                continue
            try:
                ipaddress.ip_address(candidate)
                return candidate
            except ValueError:
                continue

    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        try:
            ipaddress.ip_address(real_ip)
            return real_ip
        except ValueError:
            pass
    return direct_ip


def _new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def _log_exception(context: str, request_id: str) -> None:
    logger.exception("%s failed [request_id=%s]", context, request_id)


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, coerced))


# ── Supabase setup ────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

_supabase_client = None
_warned_missing_jwt_secret = False


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
        logger.exception("Failed to initialize Supabase client")
        return None


def _extract_user_id(authorization: Optional[str]) -> Optional[str]:
    """Extract user_id from Supabase JWT. Returns None if not authenticated."""
    global _warned_missing_jwt_secret
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]

    # Primary path: ask Supabase to validate the access token and return user.
    sb = _get_supabase()
    if sb:
        try:
            user_resp = sb.auth.get_user(token)
            user_obj = getattr(user_resp, "user", None)
            if user_obj is not None:
                user_id = getattr(user_obj, "id", None)
                if user_id:
                    return user_id
            user_data = getattr(user_resp, "data", None)
            if isinstance(user_data, dict):
                nested_user = user_data.get("user")
                if isinstance(nested_user, dict) and nested_user.get("id"):
                    return nested_user["id"]
        except Exception as exc:
            logger.warning("Supabase token validation failed: %s", exc)

    # Fallback path: local JWT decode when configured.
    try:
        import jwt
        # Decode using the JWT secret from Supabase
        # The JWT secret is in Settings → API → JWT Secret
        secret = SUPABASE_JWT_SECRET
        if not secret:
            if not _warned_missing_jwt_secret:
                logger.warning("SUPABASE_JWT_SECRET is not configured; local JWT decode disabled")
                _warned_missing_jwt_secret = True
            return None
        try:
            payload = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")
        except jwt.InvalidAudienceError:
            payload = jwt.decode(token, secret, algorithms=["HS256"], options={"verify_aud": False})
        return payload.get("sub")
    except Exception as exc:
        logger.warning("Local JWT decode failed: %s", exc)
        return None


def _save_conversation(user_id: str, question: str, answer: str,
                       citations: List[str], category: Optional[str],
                       confidence: float, request_id: Optional[str] = None):
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
        logger.exception("Failed to save conversation [request_id=%s user_id=%s]", request_id or "n/a", user_id)


def _log_supabase_env_status() -> None:
    logger.info(
        "Supabase env status: url=%s service_key=%s jwt_secret=%s",
        "set" if bool(SUPABASE_URL) else "missing",
        "set" if bool(SUPABASE_SERVICE_KEY) else "missing",
        "set" if bool(SUPABASE_JWT_SECRET) else "missing",
    )


# ── Startup: preload Chroma ──────────────────────────────────────────────────
# The collection is a committed, read-only artifact, so its size never changes at
# runtime. Counting it per /health request meant a SQLite aggregate on every uptime
# ping (and /health is not rate limited).
_chunk_count: Optional[int] = None


@app.on_event("startup")
def startup_load():
    global _chunk_count
    _log_supabase_env_status()
    coll = _backend.load_collection()
    try:
        _chunk_count = coll.count()
    except Exception:
        logger.exception("failed to count chroma collection at startup")
        _chunk_count = None


# ── Schemas ───────────────────────────────────────────────────────────────────
def _sanitize_history(raw: Optional[List[Dict[str, str]]]) -> Optional[List[Dict[str, str]]]:
    """Clamp client-supplied history to the documented caps.

    Pydantic only typed this as Dict[str, str], so a client could post arbitrarily many
    multi-megabyte messages. Keep the last MAX_HISTORY_TURNS pairs, drop anything that
    isn't a well-formed role/content pair, and cap each message's length.
    """
    if not raw:
        return None
    cleaned: List[Dict[str, str]] = []
    for msg in raw[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user").strip().lower()
        if role not in {"user", "assistant"}:
            role = "user"
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content[:MAX_HISTORY_MSG_LEN]})
    return cleaned or None


class AskRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_LEN)
    history: Optional[List[Dict[str, str]]] = Field(default=None)
    # Rolling conversation summary, carried by the client between turns (the server
    # is stateless). Produced by the pipeline, echoed back in telemetry.
    conversation_summary: Optional[str] = Field(default=None)
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
    answer_mode: Optional[str] = None
    retrieval_mode: Optional[str] = None
    query_processing_degraded: bool = False
    conversation_summary: Optional[str] = None
    reranker_mode: str
    use_reranker: bool
    reranker_top_n: int
    openai_model: str
    config_source: str
    user_key_active: bool = False
    generation_provider: Optional[str] = None
    generation_model: Optional[str] = None
    triage_level: Optional[str] = None
    safety_gate_triggered: bool = False
    safety_gate_reasons: List[str] = Field(default_factory=list)
    scope_issue: Optional[str] = None
    original_question: Optional[str] = None
    context_strategy: Optional[str] = None
    fallback_reason: Optional[str] = None
    hierarchical_available: bool = False
    evidence_spans: List[Dict[str, Any]] = Field(default_factory=list)
    citation_map: Dict[str, Any] = Field(default_factory=dict)
    selected_articles: List[Dict[str, Any]] = Field(default_factory=list)
    selected_sections: List[Dict[str, Any]] = Field(default_factory=list)
    context_token_estimate: Optional[int] = None
    total_context_token_estimate: Optional[int] = None
    graph_available: bool = False
    graph_fallback_reason: Optional[str] = None
    graph_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    graph_edges: List[Dict[str, Any]] = Field(default_factory=list)
    graph_paths: List[Dict[str, Any]] = Field(default_factory=list)
    graph_supporting_spans: List[Dict[str, Any]] = Field(default_factory=list)
    graph_context_token_estimate: Optional[int] = None
    graph_context_strategy: Optional[str] = None
    graph_focus_context: bool = False
    graph_context_focused: bool = False


class MechanicsStudyRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_LEN)
    mechanics_max_items: int = Field(default=MECHANICS_MAX_ITEMS_DEFAULT)
    mechanics_include_graph: bool = True
    mechanics_include_evidence_spans: bool = True


class MechanicsStudyResponse(BaseModel):
    answer: str
    mechanics_available: bool
    mechanics_fallback_reason: Optional[str] = None
    mechanics_nerves: List[Dict[str, Any]] = Field(default_factory=list)
    mechanics_entrapment_sites: List[Dict[str, Any]] = Field(default_factory=list)
    mechanics_muscle_pairs: List[Dict[str, Any]] = Field(default_factory=list)
    mechanics_spaces: List[Dict[str, Any]] = Field(default_factory=list)
    mechanics_mechanism_chains: List[Dict[str, Any]] = Field(default_factory=list)
    mechanics_evidence_spans: List[str] = Field(default_factory=list)
    safety_gate_triggered: bool = False
    safety_gate_reasons: List[str] = Field(default_factory=list)
    scope_issue: Optional[str] = None
    original_question: Optional[str] = None


def _record_label(record: Dict[str, Any]) -> str:
    return str(
        record.get("name")
        or record.get("site_name")
        or record.get("pair_id")
        or record.get("chain_id")
        or record.get("space_id")
        or "mechanics record"
    )


def _unique_strings(values: List[Any], limit: int = 20) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _collect_evidence_span_ids(ctx: Dict[str, Any]) -> List[str]:
    span_ids: List[Any] = []
    for key in ("nerves", "entrapment_sites", "muscle_pairs", "spaces", "mechanism_chains"):
        for record in ctx.get(key, []) or []:
            for span_key in ("evidence_span_ids", "direct_support_span_ids", "indirect_support_span_ids"):
                span_ids.extend(record.get(span_key, []) or [])
    return _unique_strings(span_ids, limit=40)


def _support_lines(records: List[Dict[str, Any]], *, direct: bool) -> List[str]:
    lines = []
    for record in records:
        support = str(record.get("support_level", "unknown"))
        if direct and support != "direct":
            continue
        if not direct and support == "direct":
            continue
        summary = (
            record.get("course_summary")
            or record.get("mechanical_trigger")
            or record.get("mechanical_role")
            or record.get("question_it_answers")
            or record.get("notes")
            or record.get("unsupported_or_uncertain_notes")
            or "Mechanics map record."
        )
        lines.append(f"- {_record_label(record)} ({support}): {summary}")
    return lines


def _format_list(values: List[str]) -> str:
    return ", ".join(values) if values else "None found in the current mechanics map."


def _build_mechanics_study_answer(question: str, ctx: Dict[str, Any], include_evidence_spans: bool = True) -> str:
    if not ctx.get("available"):
        reason = ctx.get("fallback_reason") or "mechanics_artifacts_unavailable"
        return (
            "**Short answer**\n"
            f"The mechanics study map is not available right now (`{reason}`), so I cannot give a mechanics-map answer.\n\n"
            "**Safety / interpretation boundary**\n"
            "This study mode is for article interpretation only. It cannot diagnose symptoms, rule out urgent problems, "
            "or prescribe treatment."
        )

    nerves = ctx.get("nerves", []) or []
    sites = ctx.get("entrapment_sites", []) or []
    pairs = ctx.get("muscle_pairs", []) or []
    spaces = ctx.get("spaces", []) or []
    chains = ctx.get("mechanism_chains", []) or []
    records = nerves + sites + pairs + spaces + chains
    span_ids = _collect_evidence_span_ids(ctx)

    structures = _unique_strings(
        [record.get("name") for record in nerves]
        + [item for site in sites for item in (site.get("nearby_muscles", []) or [])]
        + [item for site in sites for item in (site.get("nearby_bones_or_joints", []) or [])]
        + [item for chain in chains for item in (chain.get("involved_structures", []) or [])]
        + [item for chain in chains for item in (chain.get("involved_nerves_or_vessels", []) or [])]
    )
    site_names = [_record_label(site) for site in sites] + [_record_label(space) for space in spaces]
    pair_names = [", ".join(pair.get("muscles", []) or [_record_label(pair)]) for pair in pairs]
    chain_steps = []
    for chain in chains[:3]:
        steps = chain.get("steps", []) or []
        if steps:
            chain_steps.append(f"- {_record_label(chain)} ({chain.get('support_level', 'unknown')}): " + " -> ".join(steps))

    direct_lines = _support_lines(records, direct=True)
    indirect_lines = _support_lines(records, direct=False)
    uncertain_notes = _unique_strings(
        [record.get("notes") for record in nerves]
        + [site.get("unsupported_or_uncertain_notes") for site in sites]
        + [chain.get("weakest_step") for chain in chains]
        + [chain.get("safety_boundary") for chain in chains]
    )

    if direct_lines:
        matched_summary = ", ".join(structures[:3]) if structures else "matched structures"
        short = f"The current mechanics map has direct support for {matched_summary} and indirect support for mechanism chains."
    elif indirect_lines:
        short = "The current mechanics map only gives indirect or uncertain support for this question; it should be read as a study aid, not a conclusion."
    else:
        short = "The current mechanics map does not contain enough matching records to answer this beyond noting the evidence gap."

    sections = [
        "**Short answer**\n" + short,
        "**Relevant structures**\n" + _format_list(structures),
        "**Entrapment sites or mechanical spaces**\n" + _format_list(_unique_strings(site_names)),
        "**Muscle pairs/groups involved**\n" + _format_list(_unique_strings(pair_names)),
        "**Mechanism chain**\n" + ("\n".join(chain_steps) if chain_steps else "No matching mechanism chain was found in the current mechanics map."),
        "**Directly supported claims**\n" + ("\n".join(direct_lines) if direct_lines else "No directly supported matching claim was found."),
        "**Indirect or uncertain links**\n" + ("\n".join(indirect_lines) if indirect_lines else "No indirect/uncertain matching link was found."),
        "**What the corpus does not prove**\n" + (
            "\n".join(f"- {note}" for note in uncertain_notes[:8])
            if uncertain_notes
            else "- It does not prove that this mechanism explains any specific person's symptoms.\n- It does not establish a diagnosis or treatment plan."
        ),
        "**Safety / interpretation boundary**\n"
        "This mode is for learning and article interpretation only. It does not diagnose, rule conditions in or out, "
        "or prescribe treatment. New or worsening weakness/numbness, bowel or bladder changes, severe chest pain/trouble breathing, significant trauma, fever with systemic decline, or unexplained weight loss should be assessed in person urgently.",
    ]
    if include_evidence_spans:
        sections.append("**Evidence spans used**\n" + (_format_list(span_ids) if span_ids else "No evidence span IDs were available."))
    return "\n\n".join(sections)


# ── Helper: build config with safety caps ─────────────────────────────────────

# Public request-config allow-list. A user-supplied `api_key` is accepted but handled
# specially (validated, never logged, never echoed) — everything else is dropped.
_PUBLIC_CONFIG_KEYS = {"use_reranker", "reranker_top_n", "api_key", "provider", "model"}

# Format-based validation for a BYO key. Deliberately NOT tied to an "sk-" prefix
# (OpenAI uses sk-, sk-proj-, and other shapes; proxies differ). We accept a bounded
# token of key-safe characters and reject anything with whitespace/control/quote chars.
_API_KEY_RE = re.compile(r"^[A-Za-z0-9_\-]{20,512}$")

# Bounded model id: alnum plus the punctuation real model ids use (. _ - : /).
_MODEL_RE = re.compile(r"^[A-Za-z0-9._:\-\/]{1,128}$")
_KNOWN_PROVIDERS = {"openai", "groq", "cerebras", "openrouter", "mistral", "gemini"}


def _sanitize_user_api_key(value: Any) -> Optional[str]:
    """Return a cleaned BYO key if it looks structurally valid, else None.

    Never logs the value. Rejects wrong types, empty/short/overlong strings, and
    anything containing characters that don't belong in an API token.
    """
    if not isinstance(value, str):
        return None
    key = value.strip()
    if not _API_KEY_RE.match(key):
        return None
    return key


def _sanitize_provider(value: Any) -> Optional[str]:
    """Return a known generation provider name (lowercased) or None."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in _KNOWN_PROVIDERS else None


def _sanitize_model(value: Any) -> Optional[str]:
    """Return a bounded, format-valid model id, or None."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    return v if _MODEL_RE.match(v) else None


def _config_meta(cfg: QAConfig, source: str) -> Dict[str, Any]:
    return {
        "reranker_mode": "per_source" if cfg.use_reranker else "off",
        "use_reranker": bool(cfg.use_reranker),
        "reranker_top_n": int(cfg.reranker_top_n),
        "openai_model": str(cfg.openai_model),
        "config_source": source,
        # Presence flag only — the key value itself is never exposed in telemetry.
        "user_key_active": bool(getattr(cfg, "api_key", None)),
        # Requested generation selection (the model actually used is reported
        # separately from the run result, since a pinned choice can fall back).
        "generation_provider": getattr(cfg, "generation_provider", None),
        "generation_model": getattr(cfg, "generation_model", None),
    }


def _build_config(cfg_dict: Optional[Dict[str, Any]]) -> Tuple[QAConfig, Dict[str, Any]]:
    source = "default"
    ignored_keys = []
    default_cfg = {
        "num_predict": MAX_OUTPUT_TOKENS,
        "use_reranker": False,
        "reranker_top_n": 10,
    }
    if cfg_dict:
        ignored_keys = sorted(
            key for key in cfg_dict.keys() if key not in _PUBLIC_CONFIG_KEYS
        )
    if ignored_keys:
        logger.warning("Ignoring unsupported request config keys: %s", ", ".join(ignored_keys))

    safe_cfg = dict(default_cfg)
    if cfg_dict:
        if "use_reranker" in cfg_dict:
            safe_cfg["use_reranker"] = _coerce_bool(cfg_dict.get("use_reranker"), False)
        if "reranker_top_n" in cfg_dict:
            safe_cfg["reranker_top_n"] = _clamp_int(
                cfg_dict.get("reranker_top_n"),
                default=10,
                min_value=PUBLIC_RERANKER_TOP_N_MIN,
                max_value=PUBLIC_RERANKER_TOP_N_MAX,
            )
        if "api_key" in cfg_dict:
            user_key = _sanitize_user_api_key(cfg_dict.get("api_key"))
            if user_key:
                safe_cfg["api_key"] = user_key
        if "provider" in cfg_dict:
            prov = _sanitize_provider(cfg_dict.get("provider"))
            if prov:
                safe_cfg["generation_provider"] = prov
        if "model" in cfg_dict:
            mdl = _sanitize_model(cfg_dict.get("model"))
            if mdl:
                safe_cfg["generation_model"] = mdl
        if any(safe_cfg[key] != default_cfg[key] for key in ("use_reranker", "reranker_top_n")):
            source = "request_override"
        if safe_cfg.get("generation_provider") or safe_cfg.get("generation_model"):
            source = "request_override"
    if safe_cfg.get("api_key"):
        source = "user_key"

    try:
        cfg = QAConfig(**safe_cfg)
    except Exception:
        cfg = QAConfig(num_predict=MAX_OUTPUT_TOKENS, use_reranker=False, reranker_top_n=10)
        source = "default"

    return cfg, _config_meta(cfg, source)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    coll = _backend.collection
    return {
        "status": "ok",
        "chroma_loaded": coll is not None,
        "chunk_count": _chunk_count if _chunk_count is not None else 0,
    }


@app.get("/models")
def models():
    """Generation providers/models the UI dropdown may offer.

    Free providers appear as selectable only when their server-side key is
    configured; the premium provider (OpenAI) is always listed but flagged as
    requiring the user's own key. Model lists are suggestions — a custom model
    string is also accepted per provider.
    """
    try:
        return generation_catalog()
    except Exception:
        logger.exception("generation_catalog failed")
        return {"providers": [], "default_provider": "openai", "default_model": OPENAI_MODEL}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    client_ip = _client_ip(request)
    _check_rate_limit(client_ip)
    request_id = _new_request_id()

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # MAX_HISTORY_TURNS is a cap on user+assistant *pairs*, so keep 2 messages per turn.
    history = _sanitize_history(req.history)
    conversation_summary = (req.conversation_summary or "").strip()[:MAX_SUMMARY_LEN] or None
    cfg, cfg_meta = _build_config(req.config)

    try:
        res = agentic_run(question, cfg=cfg, history=history,
                          conversation_summary=conversation_summary)
    except OpenAIKeyError as exc:
        # Expected, non-crash condition: shared key missing / invalid / out of quota.
        logger.warning("ask: OpenAI key unavailable [request_id=%s] code=%s", request_id, exc.code)
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": exc.code,
                "message": "The service's API key is currently unavailable or out of quota.",
                "request_id": request_id,
            },
        )
    except Exception:
        _log_exception("ask", request_id)
        raise HTTPException(status_code=500, detail=f"Internal server error. request_id={request_id}")

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
        answer_mode=res.get("answer_mode"),
        retrieval_mode=res.get("retrieval_mode"),
        query_processing_degraded=bool(res.get("query_processing_degraded", False)),
        conversation_summary=res.get("conversation_summary"),
        reranker_mode=cfg_meta["reranker_mode"],
        use_reranker=cfg_meta["use_reranker"],
        reranker_top_n=cfg_meta["reranker_top_n"],
        openai_model=cfg_meta["openai_model"],
        config_source=cfg_meta["config_source"],
        user_key_active=bool(cfg_meta.get("user_key_active", False)),
        generation_provider=cfg_meta.get("generation_provider"),
        # The model that ACTUALLY produced the answer — null on the evidence-only path.
        # Never substitute the requested model here: a pinned provider that fails falls
        # through to evidence-only, and reporting the pin would credit a model that never ran.
        generation_model=res.get("generation_model"),
        triage_level=res.get("triage_level"),
        safety_gate_triggered=bool(res.get("safety_gate_triggered", False)),
        safety_gate_reasons=res.get("safety_gate_reasons", []) or [],
        scope_issue=res.get("scope_issue"),
        original_question=res.get("original_question") or question,
        context_strategy=res.get("context_strategy"),
        fallback_reason=res.get("fallback_reason"),
        hierarchical_available=bool(res.get("hierarchical_available", False)),
        evidence_spans=res.get("evidence_spans", []) or [],
        citation_map=res.get("citation_map", {}) or {},
        selected_articles=res.get("selected_articles", []) or [],
        selected_sections=res.get("selected_sections", []) or [],
        context_token_estimate=res.get("context_token_estimate"),
        total_context_token_estimate=res.get("total_context_token_estimate"),
        graph_available=bool(res.get("graph_available", False)),
        graph_fallback_reason=res.get("graph_fallback_reason"),
        graph_nodes=res.get("graph_nodes", []) or [],
        graph_edges=res.get("graph_edges", []) or [],
        graph_paths=res.get("graph_paths", []) or [],
        graph_supporting_spans=res.get("graph_supporting_spans", []) or [],
        graph_context_token_estimate=res.get("graph_context_token_estimate"),
        graph_context_strategy=res.get("graph_context_strategy"),
        graph_focus_context=bool(res.get("graph_focus_context", False)),
        graph_context_focused=bool(res.get("graph_context_focused", False)),
    )


@app.post("/study/mechanics", response_model=MechanicsStudyResponse)
def study_mechanics(req: MechanicsStudyRequest, request: Request):
    client_ip = _client_ip(request)
    _check_rate_limit(client_ip)
    request_id = _new_request_id()

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    red_flag_reasons = detect_red_flags(question)
    if red_flag_reasons:
        return MechanicsStudyResponse(
            answer=_red_flag_response(red_flag_reasons),
            mechanics_available=False,
            mechanics_fallback_reason="safety_gate_triggered",
            safety_gate_triggered=True,
            safety_gate_reasons=red_flag_reasons,
            original_question=question,
        )

    scope_issue = detect_scope_issue(question)
    if scope_issue:
        return MechanicsStudyResponse(
            answer=_scope_boundary_response(scope_issue),
            mechanics_available=False,
            mechanics_fallback_reason="scope_boundary",
            scope_issue=scope_issue,
            original_question=question,
        )

    max_items = _clamp_int(
        req.mechanics_max_items,
        default=MECHANICS_MAX_ITEMS_DEFAULT,
        min_value=MECHANICS_MAX_ITEMS_MIN,
        max_value=MECHANICS_MAX_ITEMS_MAX,
    )
    try:
        mechanics_ctx = build_mechanics_context(question, max_items=max_items)
    except Exception:
        _log_exception("study_mechanics", request_id)
        mechanics_ctx = {
            "available": False,
            "fallback_reason": f"mechanics_context_error. request_id={request_id}",
            "nerves": [],
            "entrapment_sites": [],
            "muscle_pairs": [],
            "spaces": [],
            "mechanism_chains": [],
        }

    if not req.mechanics_include_graph:
        mechanics_ctx["mechanism_chains"] = []
    evidence_spans = _collect_evidence_span_ids(mechanics_ctx) if req.mechanics_include_evidence_spans else []
    answer = _build_mechanics_study_answer(
        question,
        mechanics_ctx,
        include_evidence_spans=req.mechanics_include_evidence_spans,
    )
    return MechanicsStudyResponse(
        answer=answer,
        mechanics_available=bool(mechanics_ctx.get("available", False)),
        mechanics_fallback_reason=mechanics_ctx.get("fallback_reason") or None,
        mechanics_nerves=mechanics_ctx.get("nerves", []) or [],
        mechanics_entrapment_sites=mechanics_ctx.get("entrapment_sites", []) or [],
        mechanics_muscle_pairs=mechanics_ctx.get("muscle_pairs", []) or [],
        mechanics_spaces=mechanics_ctx.get("spaces", []) or [],
        mechanics_mechanism_chains=mechanics_ctx.get("mechanism_chains", []) or [],
        mechanics_evidence_spans=evidence_spans,
        original_question=question,
    )


# ── Streaming endpoint (SSE) ─────────────────────────────────────────────────
_SENTINEL = object()

STREAM_IDLE_TIMEOUT = 60    # max gap between tokens
STREAM_TOTAL_TIMEOUT = 120  # max total stream duration (the documented cap)
STREAM_JOIN_TIMEOUT = 5     # how long to wait for the worker to wind down


class _StreamCancelled(BaseException):
    """Cooperative cancellation signal for the generation worker.

    Raised from `on_token` once the SSE consumer is gone (timeout or client
    disconnect). Python cannot kill a thread, and the only hook the provider
    architecture calls on every streamed chunk is `on_token` — so that is the
    cancellation point. Raising there unwinds out of the provider's chunk loop,
    which stops consuming the response and stops the billing clock.

    Subclasses BaseException *deliberately*: qaEngine's `except Exception` handlers
    in ask_openai_llm() and generate_answer_with_fallback() must NOT swallow this,
    or a cancellation would be misread as a provider failure and would kick off the
    fallback chain — starting a brand-new generation for a client that has already
    gone away, which is the exact leak this is meant to close.
    """


def _timeout_event(request_id: str, reason: str) -> str:
    meta = {
        "error": f"{reason}. request_id={request_id}",
        "request_id": request_id,
        "complete": False,
    }
    return f"event: done\ndata: {json.dumps(meta)}\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest, request: Request,
               authorization: Optional[str] = Header(None)):
    client_ip = _client_ip(request)
    _check_rate_limit(client_ip)
    request_id = _new_request_id()

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Extract user_id from JWT (optional — guests can still use)
    user_id = _extract_user_id(authorization)

    # MAX_HISTORY_TURNS is a cap on user+assistant *pairs*, so keep 2 messages per turn.
    history = _sanitize_history(req.history)
    conversation_summary = (req.conversation_summary or "").strip()[:MAX_SUMMARY_LEN] or None
    cfg, cfg_meta = _build_config(req.config)

    token_q: queue.Queue = queue.Queue()
    result_holder: Dict[str, Any] = {}
    cancel_event = threading.Event()

    def on_token(tok: str):
        # Cancellation point. Once the consumer is gone there is nobody to receive this
        # token, so abandon the provider stream instead of generating (and paying) into
        # a queue no one reads.
        if cancel_event.is_set():
            raise _StreamCancelled()
        token_q.put(tok)

    def check_cancelled():
        if cancel_event.is_set():
            raise _StreamCancelled()

    # qaEngine uses this private hook between provider attempts. It reuses the same
    # BaseException cancellation path without changing the public generation API or
    # treating cancellation as an ordinary provider failure.
    on_token.check_cancelled = check_cancelled

    def run_engine():
        try:
            res = agentic_run(question, cfg=cfg, history=history, on_token=on_token,
                              conversation_summary=conversation_summary)
            result_holder.update(res)
        except _StreamCancelled:
            # Expected, not an error: the client went away and we unwound the provider.
            logger.info("ask_stream: generation cancelled, consumer gone [request_id=%s]", request_id)
            result_holder["cancelled"] = True
        except OpenAIKeyError as exc:
            logger.warning("ask_stream: OpenAI key unavailable [request_id=%s] code=%s", request_id, exc.code)
            result_holder["error"] = "The service's API key is currently unavailable or out of quota."
            result_holder["error_code"] = exc.code
            result_holder["request_id"] = request_id
        except Exception:
            _log_exception("ask_stream", request_id)
            result_holder["error"] = f"Internal server error. request_id={request_id}"
            result_holder["request_id"] = request_id
        finally:
            token_q.put(_SENTINEL)

    worker = threading.Thread(target=run_engine, daemon=True)
    worker.start()

    def _reap_worker():
        """Wind the generation worker down. Runs on EVERY exit path — normal completion,
        either deadline, and client disconnect (StreamingResponse closes the generator,
        so the `finally` below fires).

        On normal completion the worker has already exited and this is a no-op join.
        Otherwise it flips the cancel flag, so the worker aborts at its next token
        instead of running to completion and billing for output nobody will read.

        A worker still alive after the join is one blocked *inside* the provider call
        with no token yet delivered — cooperative cancellation cannot reach it, which is
        why the client-side request timeout (qaEngine.OPENAI_TIMEOUT_SECONDS) is the
        backstop that bounds it.
        """
        cancel_event.set()
        worker.join(timeout=STREAM_JOIN_TIMEOUT)
        if worker.is_alive():
            logger.warning(
                "ask_stream: worker still running after %ss; blocked inside the provider "
                "call, bounded by the client request timeout [request_id=%s]",
                STREAM_JOIN_TIMEOUT, request_id,
            )

    def event_stream():
        # Thin wrapper so the worker is reaped on EVERY exit: normal completion, either
        # deadline, an exception, or the client disconnecting (StreamingResponse closes
        # the generator, which raises GeneratorExit through this `finally`).
        try:
            yield from _stream_body()
        finally:
            _reap_worker()

    def _stream_body():
        # Two independent deadlines. The old code passed timeout=120 to a get() inside the
        # loop, which reset on every token — so a provider trickling one token a minute
        # could hold the worker forever and the documented 120s cap enforced nothing.
        deadline = time.monotonic() + STREAM_TOTAL_TIMEOUT
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("ask_stream exceeded total deadline [request_id=%s]", request_id)
                yield _timeout_event(request_id, "Timed out")
                return
            try:
                item = token_q.get(timeout=min(STREAM_IDLE_TIMEOUT, remaining))
            except queue.Empty:
                logger.warning("ask_stream stalled [request_id=%s]", request_id)
                yield _timeout_event(request_id, "Response stalled")
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
            "answer_mode": result_holder.get("answer_mode"),
            "retrieval_mode": result_holder.get("retrieval_mode"),
            "query_processing_degraded": bool(result_holder.get("query_processing_degraded", False)),
            "conversation_summary": result_holder.get("conversation_summary"),
            "reranker_mode": cfg_meta["reranker_mode"],
            "use_reranker": cfg_meta["use_reranker"],
            "reranker_top_n": cfg_meta["reranker_top_n"],
            "openai_model": cfg_meta["openai_model"],
            "config_source": cfg_meta["config_source"],
            "user_key_active": bool(cfg_meta.get("user_key_active", False)),
            "generation_provider": cfg_meta.get("generation_provider"),
            # Actual producing model — null for evidence-only. See /ask above.
            "generation_model": result_holder.get("generation_model"),
            "triage_level": result_holder.get("triage_level"),
            "safety_gate_triggered": bool(result_holder.get("safety_gate_triggered", False)),
            "safety_gate_reasons": result_holder.get("safety_gate_reasons", []) or [],
            "scope_issue": result_holder.get("scope_issue"),
            "original_question": result_holder.get("original_question") or question,
            "context_strategy": result_holder.get("context_strategy"),
            "fallback_reason": result_holder.get("fallback_reason"),
            "hierarchical_available": bool(result_holder.get("hierarchical_available", False)),
            "evidence_spans": result_holder.get("evidence_spans", []) or [],
            "citation_map": result_holder.get("citation_map", {}) or {},
            "selected_articles": result_holder.get("selected_articles", []) or [],
            "selected_sections": result_holder.get("selected_sections", []) or [],
            "context_token_estimate": result_holder.get("context_token_estimate"),
            "total_context_token_estimate": result_holder.get("total_context_token_estimate"),
            "graph_available": bool(result_holder.get("graph_available", False)),
            "graph_fallback_reason": result_holder.get("graph_fallback_reason"),
            "graph_nodes": result_holder.get("graph_nodes", []) or [],
            "graph_edges": result_holder.get("graph_edges", []) or [],
            "graph_paths": result_holder.get("graph_paths", []) or [],
            "graph_supporting_spans": result_holder.get("graph_supporting_spans", []) or [],
            "graph_context_token_estimate": result_holder.get("graph_context_token_estimate"),
            "graph_context_strategy": result_holder.get("graph_context_strategy"),
            "graph_focus_context": bool(result_holder.get("graph_focus_context", False)),
            "graph_context_focused": bool(result_holder.get("graph_context_focused", False)),
            "complete": "error" not in result_holder,
        }
        if "error" in result_holder:
            meta["error"] = result_holder["error"]
            meta["request_id"] = result_holder.get("request_id")
            if result_holder.get("error_code"):
                meta["error_code"] = result_holder["error_code"]
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
                    request_id,
                ),
                daemon=True,
            ).start()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── History endpoint ──────────────────────────────────────────────────────────
@app.get("/history")
def get_history(authorization: Optional[str] = Header(None),
                limit: int = Query(default=30, ge=1, le=100),
                offset: int = Query(default=0, ge=0, le=MAX_HISTORY_OFFSET)):
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
    except Exception:
        request_id = _new_request_id()
        _log_exception("history", request_id)
        raise HTTPException(status_code=500, detail=f"Internal server error. request_id={request_id}")
