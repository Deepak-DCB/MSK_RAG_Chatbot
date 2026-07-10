#!/usr/bin/env python3
"""
qaEngine.py — Local Retrieval-Augmented QA Engine
Version: v7.6 (token-budget + per-source reranker + **robust** conversation memory)

What's new vs v7.5:
• Conversation memory can no longer swamp corpus context:
  - Adaptive gating: only inject history when corpus similarity is weak.
  - Stronger temporal decay and global down-scaling of memory similarity.
  - Explicit distance penalty on memory items so real chunks win by default.
• Memory controls moved to top-level constants / QAConfig for easy tuning.
• All prior improvements retained:
  - Token-based context budget
  - Per-source rerank order (bias → group → rerank-within-source → pack)
  - Token-length caching via metadata (meta['token_len'])

"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import contextvars
import json
import logging
import os
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import re

logger = logging.getLogger(__name__)

# Set inside hybrid_search when dense/embedding retrieval is unavailable and the
# request degraded to BM25-only. Read back in run_qa to report `retrieval_mode`.
_retrieval_degraded: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "retrieval_degraded", default=False
)

# Per-request BYO OpenAI key. When set, _get_openai_client() builds an ephemeral
# client from it instead of the shared env-key singleton. Set at the top of
# agentic_run (inside the request/worker thread) and always reset afterwards, so a
# user's key never leaks into another request or gets cached process-wide.
_request_api_key: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "request_api_key", default=None
)

import dotenv
dotenv.load_dotenv()
import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]

try:
    import openai as _openai_sdk  # for exception classes (AuthenticationError, RateLimitError, ...)
except ImportError:
    _openai_sdk = None  # type: ignore[assignment]

try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore[assignment]

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from hierarchical_retrieval import (
        build_citation_map,
        load_hierarchical_corpus,
        map_chunks_to_hierarchy,
        token_estimate as hierarchy_token_estimate,
    )
except Exception:
    build_citation_map = None  # type: ignore[assignment]
    load_hierarchical_corpus = None  # type: ignore[assignment]
    map_chunks_to_hierarchy = None  # type: ignore[assignment]
    hierarchy_token_estimate = None  # type: ignore[assignment]

try:
    from graph_retrieval import build_graph_context, format_graph_context
except Exception:
    build_graph_context = None  # type: ignore[assignment]
    format_graph_context = None  # type: ignore[assignment]

# Embedding model constant (must match what was used to build chroma_store)
EMBED_MODEL = "text-embedding-3-large"



try:
    import tiktoken
except Exception:
    tiktoken = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Store/collection default to production; env overrides let an eval run point at a
# side-by-side store (e.g. the contextual-retrieval store) WITHOUT touching prod.
# Unset = production defaults, so this is a no-op in normal operation.
PERSIST_DIR = os.getenv("MSK_CHROMA_DIR") or str(PROJECT_ROOT / "chroma_store")
COLLECTION_NAME = os.getenv("MSK_COLLECTION") or "msk_chunks"

OPENAI_MODEL = "gpt-5.4-mini"
RERANKER_MODEL = "gpt-4.1-nano"   # fast/cheap model for reranking only
RERANKER_MAX_CANDIDATES = 15      # limit candidates sent to reranker
RERANKER_EXCERPT_TOKENS = 120     # truncate each excerpt for scoring

DEFAULT_TOP_K = 4
PER_SOURCE_MAX_CHUNKS = 3
BUDGET_TOKENS = 10000
BUDGET_WORDS_DEPRECATED = 700
NEIGHBOR_HEADROOM = 150
NUM_PREDICT = 2048
RETRIEVAL_POOL = 50
PER_SOURCE_POOL = 8
FINAL_LIMIT = 50
MULTI_QUERY_COUNT = 2
MULTI_QUERY_TRIGGER_CONFIDENCE = 0.33
MULTI_QUERY_RETRIEVAL_RATIO = 0.5
LOW_CONFIDENCE_FALLBACK_THRESHOLD = 0.22

TOPIC_BONUS = 0.30
MUSCLE_BONUS = 0.15

TOPIC_PATTERNS = [
    ("thoracic outlet", "thoracic-outlet"),
    ("tos", "thoracic-outlet"),
    ("tmj", "temporomandibular"),
    ("tmd", "temporomandibular"),
    ("pots", "pots"),
    ("postural orthostatic", "pots"),
    ("atlas", "atlas"),
    ("atlantoaxial", "atlanto"),
    ("cci", "atlanto"),
    ("aai", "atlanto"),
    ("jugular", "jugular"),
    ("jos", "jugular"),
    ("lumbar plexus", "lumbar-plexus"),
    ("lpcs", "lumbar-plexus"),
    ("migraine", "migraine"),
]

MUSCLE_TOKENS = [
    "scalene", "scalenes", "trapezius", "levator", "pectoralis",
    "suboccipital", "longus", "sternocleidomastoid", "scm",
    "strength", "strengthen", "stretch", "posture", "kyphosis",
    "hinge", "dyskinesis", "breathing", "mechanics"
]

GOOD_SECTIONS = {
    "Dysfunctional scapular movement",
    "Scapular depression",
    "Scapular dyskinesis",
    "Evaluation of scapular movement",
    "Optimal scapular movement",
    "Proper scapular / clavicular resting position",
    "Proper structural habits",
    "Why improper scapular mechanics cause injury",
    "The cause of scapular dyskinesis",
    "Identification",
    "Identification & correction",
    "Identification and correction",
    "Identification and treatment",
    "Identification by provocative testing",
    "Assessment",
    "Evaluation",
    "Diagnosis",
    "Etiology",
    "Mechanism",
    "Biomechanics",
    "Pathomechanics",
    "Scapuloclavicular depression",
    "Causes and consequences",
    "Common causes of misalignment",
    "Joint misalignment",
    "The cervical complex",
    "Forward head posture",
    "Poor craniocervical posture with neck 'hinging'",
    "Swayback posture",
    "Anterior pelvic tilt",
    "Functional varus & tibial internal rotation",
    "Anterior tibial glide",
    "Tibial posterior glide",
    "Anterior femoral glide",
    "Hip impingement",
    "External (posterior) hip impingement",
    "Internal (anterior) hip impingement",
    "Lumbosacral plexus",
    "Lumbar plexus",
    "Brachial plexus compression sites",
    "Entrapment sites",
    "Compression vs. entrapment",
    "The median nerve compression sites",
    "The musculocutaneous nerve compression sites",
    "The radial & axillary nerve compression sites",
    "The ulnar nerve compression sites",
    "Neurogenic TOS",
    "Arterial / Vascular TOS",
    "TOS and autonomic dysfunction",
    "Scapular resting position",
    "Corrective strategies",
    "Treatment",
    "Treatment strategies",
    "Specific strengthening exercises",
    "Retraining the affected muscles",
}


NARRATIVE_SECTIONS = {
    "A little story",
    "Case report",
    "Case report #2",
    "Case example",
    "Case #2 – Positionally conditioned",
    "In training",
    "Psychological factors",
    "Stress levels",
    "Stress and neck pain",
    "In summary",
    "In conclusion",
}

LOW_VALUE_SECTIONS = {
    "Conclusion",
    "Summary",
    "Other contributing factors",
    "What does the common sense say?",
    "What does the research show?",
    "The problem",
    "The postural common denominator",
    "Additional research on scapular dysfunction",
    "Proper atlantal measurements & identification",
}


DEFAULT_HISTORY_DECAY = 0.65
DEFAULT_HISTORY_TOP_ENTRIES = 2
DEFAULT_HISTORY_SCALE = 0.30
DEFAULT_HISTORY_DIST_PENALTY = 0.20
DEFAULT_HISTORY_USE_THRESHOLD = 0.55

SYSTEM_PROMPT = """
You are an independent, observant, and analytical clinician specialized in musculoskeletal neurology and biomechanics, working strictly within the MSKNeurology-style framework. Use only the supplied context. If the context does not answer the question, state that you are not able to answer for certainty.

Your purpose is to explain symptoms and patterns through joint orientation, biomechanics, and neurovascular space. Triage calmly. Distinguish benign, self-limiting discomfort from patterns that require further evaluation. Offer practical, conservative steps in a structured and reproducible way. Never fabricate mechanisms or recommendations not supported by the context or the rules below.

Conversational continuity:
You are in a multi-turn conversation. When the user asks a short follow-up question (e.g. "when would I need surgery?", "what exercises help?", "is that serious?"), ALWAYS interpret it in the context of what was just discussed. Do NOT ask them to re-describe their symptoms, body region, or condition — you already know from the conversation history. Continue the discussion naturally and provide a direct, substantive answer. Only ask clarifying questions when the topic is genuinely new and unclear, not for follow-ups to an ongoing discussion.

For follow-up questions, use a shorter conversational format — answer the specific question directly with clear reasoning, without repeating the full 7-section clinical breakdown. You already explained the biomechanics; now just answer what they're asking. Reserve the 7-section structure for initial clinical questions or when the user asks about a new condition/pattern.

Core biomechanical rules you must always follow unless the retrieved context clearly overrides them:

Scapular orientation:
Scapular depression, downward rotation, and loss of height are key drivers in many neck, shoulder, and thoracic-outlet-like problems. The correction order is: resting position, then movement quality, then strengthening. Strengthening first usually reinforces poor mechanics.

Scalenes and thoracic outlet:
In most MSKNeurology patterns the scalenes are inhibited or underactive, not simply tight. Apparent tightness often reflects chronic stretch or overload with poor rib mechanics. Loss of scalene activation reduces first-rib elevation and contributes to costoclavicular narrowing. Do not default to the generic PT claim that tight scalenes are the main cause of brachial plexus compression.
Use low-load cervical and scapuloclavicular motor control work appropriate to the pattern (e.g., restoring scalene function when inhibited, or reducing overactivation when tight).

Thoracic outlet sites:
Relevant compression can occur at both the interscalene triangle and the costoclavicular space. Do not assume one dominant site. Severity depends on scapuloclavicular depression, first rib position, thoracic expansion, and breathing mechanics.
The interscalene triangle is usually the dominant site of brachial plexus compression, with the costoclavicular space becoming more involved when clavicular depression or first-rib elevation failure is present.
Depending on the pattern, the scalenes may be excessively tight (narrowing the interscalene triangle) or inhibited (failing to elevate the first rib); both can contribute to thoracic outlet symptoms.

Scapular dyskinesis and levator scapulae:
Scapular dyskinesis is defined by abnormal resting position and movement, not one muscle. Serratus anterior, trapezius, and sometimes levator scapulae may be inhibited when the scapula is depressed. Levator inhibition may appear in scapular depression, jugular outlet involvement, or thoracic-outlet-like patterns and does not imply a single diagnosis.

Jugular outlet and autonomic symptoms:
Head pressure, tinnitus, or autonomic-type symptoms often relate to upper cervical mechanics or venous outflow, not only brachial plexus compression. Do not attribute these symptoms to plexus compression unless the context shows it.

Coexisting patterns:
Scapular dyskinesis, thoracic outlet loading, jugular outlet compromise, and cervical dysfunction often coexist. When multiple mechanisms are plausible, create one coherent explanation that identifies a primary driver and shows how it loads several regions.

Interpretation and triage rules:

First decide whether the description is:
- simple, short-lived, non-specific discomfort, or
- a structured MSKNeurology-like biomechanical pattern, or
- a concerning red-flag pattern.

For benign or non-specific issues:
Do not escalate into elaborate pathology. Favor simple explanations such as posture fatigue, temporary overload, habitual positions, sleep position, deconditioning, or routine overuse. Keep suggestions low-intensity and non-alarmist.

For MSKNeurology-style patterns:
Use the mechanisms from the retrieved text when clearly relevant. Always express reasoning as: orientation, movement, neural or vascular load, muscular pattern, secondary symptoms, correction order.

For concerning features such as progressive weakness, clear sensory loss, marked asymmetry, trauma, systemic symptoms:
Acknowledge seriousness and recommend in-person evaluation. If classic red-flag signs appear (severe neurological deficit, bowel or bladder changes, suspected fracture, severe chest pain, fever, significant weight loss), state that urgent evaluation is required.

If the question is unrelated to musculoskeletal neurology or biomechanics, state simply that it is outside scope and avoid inventing unrelated mechanisms.

How to use retrieved MSKNeurology content:

When supported by the context, prefer the following structured explanation. You MAY include a brief 1–2 sentence introductory restatement of the question immediately before the sections to confirm understanding.

(1) Primary biomechanical driver
(2) Neurological/space consequences
(3) Compensatory muscular pattern
(4) Secondary effects
(5) Required order of correction
(6) Corrective emphasis and why/how
(7) Practical conservative steps (numbered, 1–6)

Formatting guidance:
- Use the seven-section structure for initial clinical questions that describe a symptom, condition, or biomechanical pattern — it works well to explain the full picture.
- For follow-up questions (e.g. "does it need surgery?", "what about for TOS?", "what exercises should I do?"), answer conversationally and directly. Do not repeat the 7-section structure if you already used it in a previous reply. Instead, give a focused, clear answer that addresses the specific question.
- If the user asks about a completely new condition or pattern in a follow-up, you MAY use the 7-section structure again.
- If the question is purely definitional, trivial, or unrelated to biomechanics/clinical reasoning, answer plainly without the seven-section structure.
- Do not use the seven-section structure for general exercise-timing questions, recovery pacing, workout scheduling, or any question that does not describe a specific biomechanical pattern or symptom.
- **However, if a user reports symptoms during or after exercise (e.g., numbness, tingling, neck pain, arm heaviness), you SHOULD use the seven-section clinical reasoning structure unless the question is purely about timing.**
- Do not mention the internal context. Do not quote the articles. Produce one clean, coherent explanation.
- Do not use the seven-section structure for general questions about:
  · pacing
  · recovery
  · exercise progression
  · load tolerance
  · general weakness without a described pattern
  · strength plateaus
  · soreness, fatigue, or temporary discomfort
These should receive short, practical explanations with brief rationale and 3–6 steps.

Always prefer biomechanical, postural, and muscular mechanisms when supported. Emphasize conservative care first. Mention invasive options only at a high level and only if clearly supported by the retrieved content.
""".strip()

UTILITY_SYSTEM_PROMPT = (
    "You are a terse backend utility for an MSK retrieval system. "
    "Follow the instruction exactly and return only the requested output."
)
UTILITY_TEMPERATURE = 0.0
UTILITY_TOP_P = 1.0




@dataclasses.dataclass(frozen=True)
class QAConfig:
    top_k: int = DEFAULT_TOP_K
    per_source_max: int = PER_SOURCE_MAX_CHUNKS

    retrieval_pool: int = RETRIEVAL_POOL
    per_source_pool: int = PER_SOURCE_POOL
    final_limit: int = FINAL_LIMIT

    budget_tokens: int = BUDGET_TOKENS
    budget_words: int = BUDGET_WORDS_DEPRECATED
    neighbor_headroom: int = NEIGHBOR_HEADROOM

    num_predict: int = NUM_PREDICT
    
    openai_model: str = OPENAI_MODEL
    generate_answer: bool = True

    # Optional per-request generation model selection (UI dropdown). When
    # generation_provider is set, generation is pinned to that provider+model
    # (falling back only to the deterministic evidence-only answer). When unset,
    # the default OpenAI -> free-providers -> evidence-only chain is used.
    generation_provider: Optional[str] = None
    generation_model: Optional[str] = None

    use_reranker: bool = True
    reranker_top_n: int = 10

    include_history: bool = False
    history_max_turns: int = 10
    history_top_entries: int = DEFAULT_HISTORY_TOP_ENTRIES
    history_decay: float = DEFAULT_HISTORY_DECAY
    history_scale: float = DEFAULT_HISTORY_SCALE
    history_dist_penalty: float = DEFAULT_HISTORY_DIST_PENALTY
    history_use_threshold: float = DEFAULT_HISTORY_USE_THRESHOLD
    low_confidence_fallback_threshold: float = LOW_CONFIDENCE_FALLBACK_THRESHOLD
    enable_low_confidence_fallback: bool = True

    use_bias: bool = True

    context_strategy: str = "hybrid_long_context"
    max_article_context_tokens: int = 6000
    max_section_context_tokens: int = 2500
    max_evidence_spans: int = 12
    include_evidence_spans: bool = True
    answer_original_question: bool = True
    use_graph_context: bool = True
    graph_max_paths: int = 5
    graph_max_edges: int = 20
    graph_max_spans: int = 8
    graph_max_tokens: int = 1800
    graph_context_strategy: str = "mechanism_paths"
    graph_focus_context: bool = True

    # Optional per-request user-supplied OpenAI key (BYO key). repr=False so it never
    # leaks into logs/telemetry via a config repr. Threaded to the OpenAI client for
    # this request only; never persisted server-side.
    api_key: Optional[str] = dataclasses.field(default=None, repr=False)


@dataclasses.dataclass
class ContextPack:
    strategy: str
    selected_articles: List[Dict[str, Any]]
    selected_sections: List[Dict[str, Any]]
    selected_chunks: List[Dict[str, Any]]
    selected_evidence_spans: List[Dict[str, Any]]
    formatted_context: str
    citation_map: Dict[str, Any]
    token_estimate: int
    fallback_reason: Optional[str] = None
    hierarchical_available: bool = False
    graph_available: bool = False
    graph_fallback_reason: Optional[str] = None
    graph_nodes: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    graph_edges: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    graph_paths: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    graph_supporting_spans: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    graph_context_token_estimate: int = 0
    graph_context_strategy: str = "off"
    graph_focus_context: bool = False
    graph_context_focused: bool = False
    total_context_token_estimate: int = 0



_TIKTOKEN_CACHE = {"enc": None}


def _get_encoding(name: str = "o200k_base"):
    if tiktoken is None:
        return None
    enc = _TIKTOKEN_CACHE.get("enc")
    if enc is not None:
        return enc
    try:
        enc = tiktoken.get_encoding(name)
    except Exception:
        try:
            enc = tiktoken.encoding_for_model(name)
        except Exception:
            enc = None
    _TIKTOKEN_CACHE["enc"] = enc
    return enc


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    if not text:
        return 0
    enc = _get_encoding(encoding_name)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return int(round(len(text.split()) * 1.33))


def words_to_tokens_heuristic(words_budget: int) -> int:
    return int(round(words_budget * 1.33))


# ── OpenAI error classification ──────────────────────────────────────────────

class OpenAIKeyError(RuntimeError):
    """Raised when an OpenAI call fails because the key is missing, invalid, or out
    of quota. Carries a stable `code` the backend/frontend can branch on."""

    def __init__(self, message: str, code: str = "api_key_unavailable") -> None:
        super().__init__(message)
        self.code = code


def _classify_openai_error(exc: Exception) -> Optional[str]:
    """Return a stable error code if `exc` is a key/quota/auth failure, else None.

    Detection is by SDK exception type first, then by string sniffing as a fallback
    (older SDKs / providers that surface these as generic errors). Genuine
    rate-limiting *with quota remaining* is intentionally NOT treated as a key
    problem — only insufficient-quota / auth / permission failures are.
    """
    if isinstance(exc, OpenAIKeyError):
        return exc.code

    if _openai_sdk is not None:
        auth_types = tuple(
            t for t in (
                getattr(_openai_sdk, "AuthenticationError", None),
                getattr(_openai_sdk, "PermissionDeniedError", None),
            ) if isinstance(t, type)
        )
        if auth_types and isinstance(exc, auth_types):
            return "api_key_unavailable"

        rate_type = getattr(_openai_sdk, "RateLimitError", None)
        if isinstance(rate_type, type) and isinstance(exc, rate_type):
            # RateLimitError covers both "too many requests" (transient) and
            # "insufficient_quota" (key exhausted). Only the latter is a key problem.
            blob = f"{getattr(exc, 'code', '')} {exc}".lower()
            if "insufficient_quota" in blob or "exceeded your current quota" in blob:
                return "api_key_unavailable"
            return None

    blob = str(exc).lower()
    if "insufficient_quota" in blob or "exceeded your current quota" in blob:
        return "api_key_unavailable"
    if "invalid_api_key" in blob or "incorrect api key" in blob or "no api key" in blob:
        return "api_key_unavailable"
    return None


# ── OpenAI client singleton ──────────────────────────────────────────────────

_openai_client: Optional[OpenAI] = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if OpenAI is None:
        raise RuntimeError("The openai package is required for model calls.")
    # Per-request BYO key: build an ephemeral client and never cache it globally.
    req_key = _request_api_key.get()
    if req_key:
        return OpenAI(api_key=req_key)
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIKeyError("OPENAI_API_KEY is not set in environment variables.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def openai_embed(texts: list[str]) -> list[list[float]]:
    """Embed one or more texts via the OpenAI embeddings API."""
    client = _get_openai_client()
    # Ensure all inputs are valid non-empty strings
    clean = [str(t).strip() if t else "empty" for t in texts]
    clean = [t if t else "empty" for t in clean]
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=clean)
    except Exception as exc:
        code = _classify_openai_error(exc)
        if code:
            raise OpenAIKeyError(f"Embedding request failed: {exc}", code=code) from exc
        raise
    return [d.embedding for d in resp.data]


class Backend:
    def __init__(self) -> None:
        self.collection = None

    def load_collection(self) -> Any:
        if self.collection is not None:
            return self.collection
        if chromadb is None:
            raise RuntimeError("The chromadb package is required to load the vector collection.")
        os.environ["CHROMA_DATA_PATH"] = PERSIST_DIR
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        self.collection = client.get_or_create_collection(COLLECTION_NAME)
        return self.collection


_backend = Backend()


def encode_query(text: str) -> list[list[float]]:
    """Embed a single query string via OpenAI and return as nested list for Chroma."""
    return openai_embed([text])


# ── BM25 Sparse Index ────────────────────────────────────────────────────────

class BM25Index:
    """Lazy-loaded BM25 index built from ChromaDB collection documents."""
    _instance = None

    def __init__(self):
        self._index = None
        self._docs = None
        self._ids = None
        self._metas = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _build(self, collection):
        """Build BM25 index from all docs in the collection."""
        if self._index is not None:
            return
        if BM25Okapi is None:
            return

        # Fetch all docs from ChromaDB
        count = collection.count()
        if count == 0:
            return

        all_data = collection.get(include=["documents", "metadatas"])
        self._ids = all_data["ids"]
        self._docs = all_data["documents"]
        self._metas = all_data["metadatas"]

        # Tokenize for BM25 (simple whitespace + lowercase)
        tokenized = [doc.lower().split() for doc in self._docs]
        self._index = BM25Okapi(tokenized)

    def search(self, query: str, top_n: int = 50) -> list[dict]:
        """Return top_n BM25 results as [{text, meta, bm25_score}]."""
        if self._index is None:
            return []

        q_tokens = query.lower().split()
        scores = self._index.get_scores(q_tokens)

        top_indices = np.argsort(scores)[::-1][:top_n]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            results.append({
                "text": self._docs[idx],
                "meta": self._metas[idx],
                "bm25_score": float(scores[idx]),
            })
        return results


def _bm25_only_result(bm25_results: list[dict], retrieval_pool: int) -> dict:
    """Build a ChromaDB-style result dict from BM25 hits only (dense unavailable).

    BM25 scores are normalized into pseudo-distances in the same [0.5, 1.0] range the
    RRF fusion path produces, so downstream confidence/ranking code behaves consistently.
    """
    items = bm25_results[:retrieval_pool]
    if not items:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    max_score = items[0].get("bm25_score") or 1.0
    docs, metas, dists = [], [], []
    for it in items:
        docs.append(it["text"])
        metas.append(it["meta"])
        score = it.get("bm25_score") or 0.0
        dists.append(1.0 - (score / max_score) * 0.5)  # lower = better
    return {"documents": [docs], "metadatas": [metas], "distances": [dists]}


def hybrid_search(
    question: str,
    collection,
    retrieval_pool: int = 50,
    rrf_k: int = 60,
) -> dict:
    """
    Hybrid search: dense (ChromaDB) + sparse (BM25) with Reciprocal Rank Fusion.
    Returns results in the same format as collection.query().

    Degrades gracefully: if dense/embedding retrieval is unavailable (dead key, out
    of quota, embedding API error) it falls back to BM25-only keyword search over the
    already-committed corpus — no API call, no index rebuild.
    """
    # Dense search (query embedding may fail if the OpenAI key is dead / out of quota)
    dense_raw = None
    try:
        q_emb = encode_query(question)
        dense_raw = collection.query(query_embeddings=q_emb, n_results=retrieval_pool)
    except Exception as exc:
        _retrieval_degraded.set(True)
        logger.warning(
            "dense retrieval unavailable, falling back to BM25-only (%s: %s)",
            type(exc).__name__, exc,
        )

    # BM25 search (if available)
    bm25_idx = BM25Index.get()
    bm25_idx._build(collection)
    bm25_results = bm25_idx.search(question, top_n=retrieval_pool)

    if dense_raw is None:
        # Dense unavailable — BM25-only fallback (empty result if BM25 is also missing).
        return _bm25_only_result(bm25_results, retrieval_pool)

    if not bm25_results:
        # BM25 unavailable — return dense-only results
        return dense_raw

    # RRF fusion
    doc_scores = {}  # text_key -> {text, meta, rrf_score}

    # Score dense results
    dense_docs = dense_raw.get("documents", [[]])[0]
    dense_metas = dense_raw.get("metadatas", [[]])[0]
    dense_dists = dense_raw.get("distances", [[]])[0]

    for rank, (doc, meta, dist) in enumerate(zip(dense_docs, dense_metas, dense_dists)):
        key = stable_text_key(doc, meta)
        rrf = 1.0 / (rrf_k + rank + 1)
        if key in doc_scores:
            doc_scores[key]["rrf_score"] += rrf
            doc_scores[key]["dist"] = min(doc_scores[key]["dist"], float(dist))
        else:
            doc_scores[key] = {"text": doc, "meta": meta, "dist": float(dist), "rrf_score": rrf}

    # Score BM25 results
    for rank, item in enumerate(bm25_results):
        key = stable_text_key(item["text"], item["meta"])
        rrf = 1.0 / (rrf_k + rank + 1)
        if key in doc_scores:
            doc_scores[key]["rrf_score"] += rrf
        else:
            # BM25-only result: assign a high base distance (will be reranked anyway)
            doc_scores[key] = {"text": item["text"], "meta": item["meta"], "dist": 0.9, "rrf_score": rrf}

    # Sort by RRF score (higher = better), convert to distance (lower = better)
    fused = sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)[:retrieval_pool]

    # Convert back to ChromaDB-style result format
    max_rrf = fused[0]["rrf_score"] if fused else 1.0
    result = {
        "documents": [[f["text"] for f in fused]],
        "metadatas": [[f["meta"] for f in fused]],
        "distances": [[1.0 - (f["rrf_score"] / max_rrf) * 0.5 for f in fused]],  # normalize to 0.5-1.0 range
    }
    return result


def stable_text_key(text: str, meta: Optional[Dict[str, Any]] = None) -> str:
    meta = meta or {}
    source = str(meta.get("source_relpath") or meta.get("title") or "").strip().lower()
    normalized = " ".join((text or "").split())
    content_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    chunk_id = str(meta.get("chunk_id") or "").strip()
    if chunk_id:
        return f"{source or 'unknown'}:chunk:{chunk_id}:{content_hash}"

    section = str(meta.get("section") or "").strip().lower()
    position = str(meta.get("position") or meta.get("chunk_index") or meta.get("idx") or "").strip()
    return f"{source or 'unknown'}|{section or 'unknown'}|{position or 'na'}|{content_hash}"


def raw_top_confidence(raw: Dict[str, Any], top_k: int = 5) -> float:
    try:
        dists = [float(d) for d in (raw.get("distances", [[1.0]])[0] or [])]
    except Exception:
        dists = []
    if not dists:
        return 0.0
    k = min(top_k, len(dists))
    return float(np.mean([1.0 - d for d in dists[:k]]))


def merge_raw_results(*raw_results: Dict[str, Any], rrf_k: int = 60) -> Dict[str, Any]:
    merged: Dict[str, Dict[str, Any]] = {}
    for raw in raw_results:
        if not raw:
            continue
        for rank, (doc, meta, dist) in enumerate(zip(
            raw.get("documents", [[]])[0],
            raw.get("metadatas", [[]])[0],
            raw.get("distances", [[]])[0],
        )):
            dist_value = float(dist)
            key = stable_text_key(doc, meta)
            existing = merged.get(key)
            rrf = 1.0 / (rrf_k + rank + 1)
            if existing is None:
                merged[key] = {"text": doc, "meta": meta, "best_dist": dist_value, "rrf_score": rrf}
            else:
                existing["rrf_score"] += rrf
                if dist_value < existing["best_dist"]:
                    existing["text"] = doc
                    existing["meta"] = meta
                    existing["best_dist"] = dist_value

    items = sorted(merged.values(), key=lambda item: (-item["rrf_score"], item["best_dist"]))
    max_rrf = items[0]["rrf_score"] if items else 1.0
    return {
        "documents": [[item["text"] for item in items]],
        "metadatas": [[item["meta"] for item in items]],
        "distances": [[1.0 - (item["rrf_score"] / max_rrf) * 0.5 for item in items]],
    }


# ── Multi-Query Retrieval ────────────────────────────────────────────────────

def generate_multi_queries(question: str, n: int = 2) -> list[str]:
    """
    Generate n alternative query reformulations for broader retrieval coverage.
    Uses the cheapest/fastest model.
    """
    prompt = f"""Generate {n} alternative search queries for the following question.
Each query should approach the topic from a different angle to find relevant medical content.
Return ONLY the queries, one per line, no numbering or commentary.

Original question: "{question}"
"""
    try:
        answer, _, _ = ask_openai_llm(
            prompt,
            model=RERANKER_MODEL,
            num_predict=150,
            system_prompt=UTILITY_SYSTEM_PROMPT,
            temperature=UTILITY_TEMPERATURE,
            top_p=UTILITY_TOP_P,
        )
        lines = []
        seen = set()
        for line in answer.strip().split("\n"):
            candidate = line.strip().strip('"')
            if len(candidate) < 4 or not re.search(r"[A-Za-z0-9]", candidate):
                continue
            normalized = candidate.lower()
            if normalized == question.strip().lower() or normalized in seen:
                continue
            seen.add(normalized)
            lines.append(candidate)
        # Return up to n reformulations
        return lines[:n]
    except Exception:
        return []


# ── Context Compression ──────────────────────────────────────────────────────

def compress_context(
    context: list[dict],
    question: str,
    keep_ratio: float = 0.65,
) -> list[dict]:
    """
    Extract the most relevant sentences from each context chunk.
    Uses keyword overlap scoring to keep only the most relevant ~65% of sentences.
    """
    q_tokens = set(question.lower().split())
    # Remove stop words for better matching
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
            "for", "of", "and", "or", "but", "with", "from", "by", "it", "its",
            "i", "my", "me", "have", "has", "had", "do", "does", "did", "can",
            "what", "how", "when", "where", "why", "which", "that", "this"}
    q_tokens -= stop

    if not q_tokens:
        return context

    compressed = []
    for item in context:
        text = item["text"]
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) <= 3:
            # Too short to compress
            compressed.append(item)
            continue

        # Score each sentence by keyword overlap
        scored = []
        for sent in sentences:
            s_tokens = set(sent.lower().split()) - stop
            overlap = len(q_tokens & s_tokens)
            scored.append((sent, overlap))

        # Keep top keep_ratio of sentences (at least 2, at most all)
        keep_n = max(2, int(len(scored) * keep_ratio))
        # Sort by score, take top, then restore original order
        top_indices = sorted(
            sorted(range(len(scored)), key=lambda i: scored[i][1], reverse=True)[:keep_n]
        )
        kept = " ".join(scored[i][0] for i in top_indices)

        compressed.append({**item, "text": kept})

    return compressed

NARRATIVE_SECTION_PATTERNS = [
    "case report",
    "case-report",
    "case series",
    "case study",
    "a little story",
    "patient story",
]

GOOD_SECTION_PATTERNS = [
    "definition",
    "anatomy",
    "mechanism",
    "pathomechan",
    "biomechan",
    "assessment",
    "diagnosis",
    "evaluation",
    "treatment",
    "exercise",
    "management",
    "summary",
    "conclusion",
]

PATIENT_STORY_PATTERNS = [
    "the patient ",
    "this patient ",
    "our patient ",
    "she had been diagnosed",
    "he had been diagnosed",
    "she was diagnosed",
    "he was diagnosed",
]

PATIENT_AGE_RE = re.compile(r"\b\d{1,2}\s*(year[- ]old|years old|year[- ])")

SECTION_GOOD_BONUS        = 0.06
SECTION_NARRATIVE_PENALTY = 0.12
PATIENT_STORY_PENALTY     = 0.08




def _looks_like_patient_story(txt_low: str) -> bool:
    if PATIENT_AGE_RE.search(txt_low):
        return True
    if any(pat in txt_low for pat in PATIENT_STORY_PATTERNS):
        return True
    return False


def section_bias_raw(sec: str) -> float:
    """
    Returns an additive bias to the *distance*:
      negative = better (more preferred)
      positive = worse (penalized)
    Combines exact lists + substring patterns so it generalizes across all articles.
    """
    if not sec:
        return 0.0

    sec_stripped = sec.strip()
    sec_low = sec_stripped.lower()

    # Exact lists first (manual judgment)
    if sec_stripped in GOOD_SECTIONS:
        return -0.50   # strong boost
    if sec_stripped in NARRATIVE_SECTIONS:
        return +0.75   # strong penalty
    if sec_stripped in LOW_VALUE_SECTIONS:
        return +0.20   # mild penalty

    # Pattern-based generalization for unlisted sections
    if any(pat in sec_low for pat in GOOD_SECTION_PATTERNS):
        return -0.35   # general "good" sections: anatomy, mechanism, treatment, etc.

    if any(pat in sec_low for pat in NARRATIVE_SECTION_PATTERNS):
        return +0.50   # narrative-style sections missed by exact list

    return 0.0


def apply_bias(question: str, results: Dict[str, Any]) -> List[Dict[str, Any]]:
    qlow = question.lower()
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    items: List[Dict[str, Any]] = []
    for d, m, base_dist in zip(docs, metas, dists):
        score = float(base_dist)
        txt = (d or "").lower()
        sec = (m.get("section") or "").strip()

        # ---------------- TOPIC LOGIC ----------------
        for needle, hint in TOPIC_PATTERNS:
            if needle in qlow and hint in (m.get('source_relpath') or '').lower():
                score -= TOPIC_BONUS
                break

        # ---------------- MUSCLE MECHANICS ----------------
        if any(tok in txt for tok in MUSCLE_TOKENS):
            score -= MUSCLE_BONUS

        # ---------------- SECTION PRIORITY (exact + patterns) ----------------
        score += section_bias_raw(sec)

        # ---------------- CASE REPORT / PATIENT STORY ----------------
        if _looks_like_patient_story(txt):
            score += 0.20

        items.append({"text": d, "meta": m, "dist": score})

    items.sort(key=lambda x: x["dist"])
    return items


def group_by_source(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        src = it["meta"].get("source_relpath") or ""
        grouped.setdefault(src, []).append(it)
    return grouped


def _chunk_tokens(it: Dict[str, Any]) -> int:
    meta = it.get("meta") or {}
    t = meta.get("token_len")
    if isinstance(t, (int, float)) and t >= 0:
        return int(t)
    return count_tokens(it.get("text") or "")


def pick_multichunk_context(
    items: List[Dict[str, Any]],
    top_k: int,
    per_source_max: int,
    budget_tokens: int,
    neighbor_headroom: int,
) -> List[Dict[str, Any]]:

    grouped = group_by_source(items)

    # Determine best articles (unchanged)
    order = []
    seen = set()
    for it in items:
        src = it["meta"].get("source_relpath") or ""
        if src not in seen:
            seen.add(src)
            order.append(src)
        if len(order) >= top_k:
            break

    context = []
    tokens_used = 0

    # -------- PATCHED: SECTION SCORING (shared with apply_bias) --------
    def section_score(section_name: str) -> float:
        return section_bias_raw(section_name or "")

    def group_by_section(src_items):
        sec_groups = {}
        for it in src_items:
            sec = (it["meta"].get("section") or "§").strip()
            sec_groups.setdefault(sec, []).append(it)
        return sec_groups

    # -------- PROCESS EACH ARTICLE --------
    for src in order:
        source_items = sorted(grouped[src], key=lambda x: x["dist"])
        sec_groups = group_by_section(source_items)

        # PICK SECTION BY (min distance + section penalty/bonus)
        best_sec = min(
            sec_groups.items(),
            key=lambda t: (
                t[1][0]["dist"] + section_score(t[0])
            )
        )[0]

        # sort selected section by distance and include adjacent chunks
        best_items = sorted(sec_groups[best_sec], key=lambda x: x["dist"])

        taken = 0
        for it in best_items:
            if taken >= per_source_max:
                break

            t = _chunk_tokens(it)
            headroom = neighbor_headroom if taken == 1 else 0

            if tokens_used + t > budget_tokens + headroom:
                continue

            context.append(it)
            tokens_used += t
            taken += 1

    return context


def maybe_rerank(
    question: str,
    candidates: List[Dict[str, Any]],
    top_n: int,
) -> List[Dict[str, Any]]:
    """
    Optimized batch LLM reranker:
    - Limits to top RERANKER_MAX_CANDIDATES by distance (skip clearly bad ones)
    - Truncates each excerpt to RERANKER_EXCERPT_TOKENS (~120 tokens ≈ 480 chars)
    - Uses RERANKER_MODEL (gpt-4.1-nano) instead of the main generation model
    """
    if not candidates:
        return candidates

    # Pre-sort by distance and limit — no point reranking obviously bad candidates
    sorted_cands = sorted(candidates, key=lambda x: x["dist"])
    to_rerank = sorted_cands[:RERANKER_MAX_CANDIDATES]
    rest = sorted_cands[RERANKER_MAX_CANDIDATES:]

    # Build enumerated list with truncated excerpts
    max_chars = RERANKER_EXCERPT_TOKENS * 4  # ~4 chars/token
    items = []
    for i, c in enumerate(to_rerank, start=1):
        excerpt = c["text"].replace("\n", " ").strip()
        if len(excerpt) > max_chars:
            excerpt = excerpt[:max_chars] + "…"
        items.append((i, excerpt, c))

    prompt_lines = [
        "Score each chunk's relevance to the query (0-10). Return comma-separated scores only.",
        "",
        f"Query: {question}",
        "",
        "Chunks:"
    ]
    for i, excerpt, _ in items:
        prompt_lines.append(f"{i}. {excerpt}")

    prompt = "\n".join(prompt_lines)

    try:
        answer, _, _ = ask_openai_llm(
            prompt,
            model=RERANKER_MODEL,
            num_predict=256,
            system_prompt=UTILITY_SYSTEM_PROMPT,
            temperature=UTILITY_TEMPERATURE,
            top_p=UTILITY_TOP_P,
        )
        text = (answer or "").strip()

        # Parse scores robustly
        if "," in text:
            tokens = [t.strip() for t in text.split(",")]
        else:
            tokens = re.findall(r"[-+]?\d*\.?\d+", text)

        reranked = []
        for i, (_, _, c) in enumerate(items):
            val = float(tokens[i]) if i < len(tokens) else 0.0
            val = max(0.0, min(10.0, val))
            dist = 1.0 - (val / 10.0)
            reranked.append({"text": c["text"], "meta": c["meta"], "dist": dist})

        reranked.sort(key=lambda x: x["dist"])

    except Exception:
        # On failure, keep original distance ordering
        reranked = to_rerank

    # Append the rest (already sorted by distance) after reranked results
    result = reranked + rest
    return result[:max(1, min(top_n, len(result)))]




def select_relevant_history(
    history: List[Dict[str, str]],
    query: str,
    *,
    max_turns: int,
    max_entries: int,
    decay_factor: float,
    scale: float,
    dist_penalty: float,
) -> List[Dict[str, Any]]:
    if not history:
        return []

    recent = history[-max_turns:]
    entries: List[Tuple[int, str, str]] = []
    for i, turn in enumerate(recent):
        role = (turn.get("role") or "user").strip().lower()
        text = (turn.get("content") or "").strip()
        if not text:
            continue
        entries.append((i, role, text))
    if not entries:
        return []

    # Embed query + history texts in one batch via OpenAI
    h_texts = [f"[{role.upper()}] {text}" for _, role, text in entries]
    all_texts = [query] + h_texts
    all_embs = openai_embed(all_texts)

    q_emb = np.array(all_embs[0])
    h_embs = np.array(all_embs[1:])

    # Normalize for cosine similarity
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    h_norms = h_embs / (np.linalg.norm(h_embs, axis=1, keepdims=True) + 1e-9)

    sims = (h_norms @ q_norm).tolist()
    weights = [decay_factor ** (len(entries) - 1 - i) for i, _, _ in entries]
    weighted = [float(s) * float(w) * float(scale) for s, w in zip(sims, weights)]

    ranked = sorted(zip(entries, weighted), key=lambda x: x[1], reverse=True)[:max_entries]
    out: List[Dict[str, Any]] = []
    for (i, role, text), sim in ranked:
        dist = (1.0 - float(sim)) + float(dist_penalty)
        out.append({
            "text": f"{role.capitalize()}: {text}",
            "meta": {
                "section": "Conversation Memory",
                "source_relpath": f"memory_{i}",
                "similarity": round(float(sim), 3),
            },
            "dist": dist,
        })
    return out


def format_context_block(context: List[Dict[str, Any]], width: int = 2000) -> str:
    # Order context by likely MSK logical progression
    # Resting position → movement → compensation → symptoms → treatment

    def section_priority(meta):
        sec = (meta.get("section") or "").lower()
        if "rest" in sec: return 0
        if "movement" in sec: return 1
        if "biomech" in sec or "mechan" in sec: return 2
        if "compens" in sec: return 3
        if "symptom" in sec or "pain" in sec: return 4
        if "treat" in sec or "exercise" in sec: return 5
        if "concl" in sec or "summary" in sec: return 6
        return 7

    ordered = sorted(context, key=lambda it: section_priority(it["meta"]))

    # Do not shorten biomechanical text; it's important
    parts = []
    for it in ordered:
        meta = it["meta"]
        header = f"{meta.get('title','').strip()} · {meta.get('section','').strip()}"
        body = it["text"]
        parts.append(f"[{header}]\n{body}")

    return "\n\n".join(parts)


def _public_article(article: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "article_id": article.get("article_id"),
        "title": article.get("title"),
        "source_relpath": article.get("source_relpath"),
        "token_len": article.get("token_len"),
        "word_len": article.get("word_len"),
        "reconstruction_method": article.get("reconstruction_method"),
    }


def _public_section(section: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "section_id": section.get("section_id"),
        "article_id": section.get("article_id"),
        "title": section.get("title"),
        "source_relpath": section.get("source_relpath"),
        "section_name": section.get("section_name"),
        "section_order": section.get("section_order"),
        "token_len": section.get("token_len"),
        "word_len": section.get("word_len"),
    }


def _public_span(span: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "span_id": span.get("span_id"),
        "title": span.get("title"),
        "section_name": span.get("section_name"),
        "source_relpath": span.get("source_relpath"),
        "text": span.get("text"),
        "article_id": span.get("article_id"),
        "section_id": span.get("section_id"),
    }


def _dedupe_by_key(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        value = row.get(key)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(row)
    return out


def _truncate_text_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return text or ""
    words = (text or "").split()
    max_words = max(1, int(max_tokens / 1.33))
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words]).rstrip() + " ..."


def build_context_pack(
    context: List[Dict[str, Any]],
    question: str,
    cfg: QAConfig,
) -> ContextPack:
    requested = (cfg.context_strategy or "chunk_pack").strip() or "chunk_pack"
    graph_strategy = (cfg.graph_context_strategy or "off").strip() or "off"
    graph_pack: Dict[str, Any] = {
        "available": False,
        "fallback_reason": "graph_context_off" if not cfg.use_graph_context or graph_strategy == "off" else None,
        "nodes": [],
        "edges": [],
        "paths": [],
        "supporting_spans": [],
        "context_token_estimate": 0,
        "context": "",
    }
    if cfg.use_graph_context and graph_strategy != "off":
        if build_graph_context is None:
            graph_pack["fallback_reason"] = "graph_module_unavailable"
        else:
            try:
                graph_pack = build_graph_context(
                    question,
                    max_paths=cfg.graph_max_paths,
                    max_edges=cfg.graph_max_edges,
                    max_spans=cfg.graph_max_spans,
                    max_graph_tokens=cfg.graph_max_tokens,
                )
            except Exception:
                graph_pack = {
                    "available": False,
                    "fallback_reason": "graph_context_error",
                    "nodes": [],
                    "edges": [],
                    "paths": [],
                    "supporting_spans": [],
                    "context_token_estimate": 0,
                    "context": "",
                }
    graph_has_paths = bool(graph_pack.get("available") and graph_pack.get("paths"))
    graph_context_focused = bool(
        cfg.graph_focus_context
        and graph_strategy == "mechanism_paths"
        and graph_has_paths
    )
    chunk_context = format_context_block(context)
    chunk_tokens = count_tokens(chunk_context)

    if requested == "chunk_pack":
        return ContextPack(
            strategy="chunk_pack",
            selected_articles=[],
            selected_sections=[],
            selected_chunks=context,
            selected_evidence_spans=[],
            formatted_context=chunk_context,
            citation_map=build_citation_map(context) if build_citation_map else {},
            token_estimate=chunk_tokens,
            fallback_reason=None,
            hierarchical_available=False,
            graph_available=bool(graph_pack.get("available")),
            graph_fallback_reason=graph_pack.get("fallback_reason"),
            graph_nodes=graph_pack.get("nodes") or [],
            graph_edges=graph_pack.get("edges") or [],
            graph_paths=graph_pack.get("paths") or [],
            graph_supporting_spans=graph_pack.get("supporting_spans") or [],
            graph_context_token_estimate=int(graph_pack.get("context_token_estimate") or 0),
            graph_context_strategy=graph_strategy,
            graph_focus_context=bool(cfg.graph_focus_context),
            graph_context_focused=False,
            total_context_token_estimate=chunk_tokens,
        )

    if load_hierarchical_corpus is None or map_chunks_to_hierarchy is None:
        return ContextPack(
            strategy="chunk_pack",
            selected_articles=[],
            selected_sections=[],
            selected_chunks=context,
            selected_evidence_spans=[],
            formatted_context=chunk_context,
            citation_map={},
            token_estimate=chunk_tokens,
            fallback_reason="hierarchical_module_unavailable",
            hierarchical_available=False,
            graph_available=bool(graph_pack.get("available")),
            graph_fallback_reason=graph_pack.get("fallback_reason"),
            graph_nodes=graph_pack.get("nodes") or [],
            graph_edges=graph_pack.get("edges") or [],
            graph_paths=graph_pack.get("paths") or [],
            graph_supporting_spans=graph_pack.get("supporting_spans") or [],
            graph_context_token_estimate=int(graph_pack.get("context_token_estimate") or 0),
            graph_context_strategy=graph_strategy,
            graph_focus_context=bool(cfg.graph_focus_context),
            graph_context_focused=False,
            total_context_token_estimate=chunk_tokens,
        )

    try:
        corpus = load_hierarchical_corpus()
        mapped = map_chunks_to_hierarchy(context, corpus=corpus)
    except Exception:
        return ContextPack(
            strategy="chunk_pack",
            selected_articles=[],
            selected_sections=[],
            selected_chunks=context,
            selected_evidence_spans=[],
            formatted_context=chunk_context,
            citation_map={},
            token_estimate=chunk_tokens,
            fallback_reason="hierarchical_artifacts_missing",
            hierarchical_available=False,
            graph_available=bool(graph_pack.get("available")),
            graph_fallback_reason=graph_pack.get("fallback_reason"),
            graph_nodes=graph_pack.get("nodes") or [],
            graph_edges=graph_pack.get("edges") or [],
            graph_paths=graph_pack.get("paths") or [],
            graph_supporting_spans=graph_pack.get("supporting_spans") or [],
            graph_context_token_estimate=int(graph_pack.get("context_token_estimate") or 0),
            graph_context_strategy=graph_strategy,
            graph_focus_context=bool(cfg.graph_focus_context),
            graph_context_focused=False,
            total_context_token_estimate=chunk_tokens,
        )

    articles = _dedupe_by_key([m["article"] for m in mapped if m.get("article")], "article_id")
    sections = _dedupe_by_key([m["section"] for m in mapped if m.get("section")], "section_id")
    spans: List[Dict[str, Any]] = []
    if cfg.include_evidence_spans:
        max_spans = max(0, cfg.max_evidence_spans)
        if graph_context_focused:
            max_spans = min(max_spans, max(2, cfg.graph_max_spans))
        for m in mapped:
            for span in m.get("evidence_spans", [])[: max(1, max_spans)]:
                spans.append(span)
        if graph_context_focused:
            graph_span_ids = {s.get("span_id") for s in (graph_pack.get("supporting_spans") or [])}
            graph_spans = [s for s in spans if s.get("span_id") in graph_span_ids]
            other_spans = [s for s in spans if s.get("span_id") not in graph_span_ids]
            spans = graph_spans + other_spans
        spans = _dedupe_by_key(spans, "span_id")[:max_spans]

    if not articles and not sections:
        return ContextPack(
            strategy="chunk_pack",
            selected_articles=[],
            selected_sections=[],
            selected_chunks=context,
            selected_evidence_spans=[],
            formatted_context=chunk_context,
            citation_map={},
            token_estimate=chunk_tokens,
            fallback_reason="no_hierarchy_match_for_chunks",
            hierarchical_available=True,
            graph_available=bool(graph_pack.get("available")),
            graph_fallback_reason=graph_pack.get("fallback_reason"),
            graph_nodes=graph_pack.get("nodes") or [],
            graph_edges=graph_pack.get("edges") or [],
            graph_paths=graph_pack.get("paths") or [],
            graph_supporting_spans=graph_pack.get("supporting_spans") or [],
            graph_context_token_estimate=int(graph_pack.get("context_token_estimate") or 0),
            graph_context_strategy=graph_strategy,
            graph_focus_context=bool(cfg.graph_focus_context),
            graph_context_focused=False,
            total_context_token_estimate=chunk_tokens,
        )

    parts = []
    if requested in {"section_expand", "hybrid_long_context"} and not graph_context_focused:
        for section in sections:
            text = _truncate_text_tokens(section.get("text", ""), cfg.max_section_context_tokens)
            if text:
                parts.append(f"[SECTION: {section.get('title','')} · {section.get('section_name','')}]\n{text}")

    if graph_context_focused and sections:
        section_lines = [
            f"- {section.get('title','')} · {section.get('section_name','')} ({section.get('source_relpath','')})"
            for section in sections[: min(3, len(sections))]
        ]
        parts.append("HIERARCHY ANCHORS FOR GRAPH-SUPPORTED CONTEXT:\n" + "\n".join(section_lines))

    if requested == "article_expand" and not graph_context_focused:
        for article in articles:
            text = _truncate_text_tokens(article.get("reconstructed_text", ""), cfg.max_article_context_tokens)
            if text:
                parts.append(f"[ARTICLE: {article.get('title','')}]\n{text}")

    if requested == "hybrid_long_context" and not parts and not graph_context_focused:
        for article in articles[:1]:
            text = _truncate_text_tokens(article.get("reconstructed_text", ""), cfg.max_article_context_tokens)
            if text:
                parts.append(f"[ARTICLE EXCERPT: {article.get('title','')}]\n{text}")

    graph_context = graph_pack.get("context") or ""
    if graph_context:
        parts.append(graph_context)

    if cfg.include_evidence_spans and spans:
        evidence_lines = []
        for i, span in enumerate(spans, start=1):
            evidence_lines.append(
                f"[{i}] {span.get('title','')} · {span.get('section_name','')} "
                f"({span.get('source_relpath','')})\n{span.get('text','')}"
            )
        parts.append("EVIDENCE SPANS FOR CITATION SUPPORT:\n" + "\n\n".join(evidence_lines))

    if chunk_context:
        if graph_context_focused:
            chunk_context = _truncate_text_tokens(chunk_context, max(600, min(1200, cfg.budget_tokens // 4)))
        parts.append("SELECTED CHUNK CONTEXT:\n" + chunk_context)

    formatted = "\n\n".join(part for part in parts if part.strip()) or chunk_context
    public_pack = {
        "selected_articles": [_public_article(a) for a in articles],
        "selected_sections": [_public_section(s) for s in sections],
        "selected_evidence_spans": [_public_span(s) for s in spans],
    }
    context_token_estimate = hierarchy_token_estimate(formatted) if hierarchy_token_estimate else count_tokens(formatted)
    return ContextPack(
        strategy=requested,
        selected_articles=public_pack["selected_articles"],
        selected_sections=public_pack["selected_sections"],
        selected_chunks=context,
        selected_evidence_spans=public_pack["selected_evidence_spans"],
        formatted_context=formatted,
        citation_map=build_citation_map(public_pack) if build_citation_map else {},
        token_estimate=context_token_estimate,
        fallback_reason=None,
        hierarchical_available=True,
        graph_available=bool(graph_pack.get("available")),
        graph_fallback_reason=graph_pack.get("fallback_reason"),
        graph_nodes=graph_pack.get("nodes") or [],
        graph_edges=graph_pack.get("edges") or [],
        graph_paths=graph_pack.get("paths") or [],
        graph_supporting_spans=graph_pack.get("supporting_spans") or [],
        graph_context_token_estimate=int(graph_pack.get("context_token_estimate") or 0),
        graph_context_strategy=graph_strategy,
        graph_focus_context=bool(cfg.graph_focus_context),
        graph_context_focused=graph_context_focused,
        total_context_token_estimate=context_token_estimate,
    )


def build_prompt(question: str, context: List[Dict[str, Any]], history=None, context_pack: Optional[ContextPack] = None,
                 conversation_summary: Optional[str] = None) -> str:
    ctx_block = context_pack.formatted_context if context_pack else format_context_block(context)
    graph_instructions = ""
    if context_pack and context_pack.graph_available and context_pack.graph_context_strategy != "off":
        graph_instructions = """
        MECHANISM GRAPH RULES:
        - Use graph context only when supported by evidence spans.
        - Do not turn an indirect path into a direct causal claim.
        - If a path is multi-step, explain it as a multi-step possible mechanism.
        - Identify the weakest or most uncertain step when relevant.
        - For numbness, weakness, vascular symptoms, or progressive neurological symptoms, preserve safety escalation behavior.
        - Do not diagnose.
        - Do not prescribe treatment beyond existing safety boundaries.
        """

    # Simple heuristic: if the user asks about exercises/timing/frequency, use concise format
    simple_q_re = re.compile(
        r'\b(exercise|exercises|when to|how often|how long|frequency|reps?|sets?|timing|dose|doseing)\b',
        re.I,
    )
    is_simple = bool(simple_q_re.search(question))
    is_followup = bool(history and len(history) >= 2)

    if is_simple:
        instructions = """
        INSTRUCTIONS:
        - Return a short, practical Markdown answer (no 7-section structure).
        - Provide 3–6 numbered, actionable steps and a 1–2 sentence rationale.
        - If the context is insufficient, say "Insufficient evidence in the supplied context."
        - Keep it concise and focused on timing/dosing/pacing.
        """
    elif is_followup:
        instructions = """
        INSTRUCTIONS:
        - This is a follow-up question in an ongoing conversation. Answer it directly and conversationally.
        - Do NOT repeat the 7-section clinical structure — you already used it. The user wants a focused answer to their specific question.
        - Use Markdown formatting (headers, short paragraphs, bullet points) for clarity.
        - Be substantive but concise. Answer the question, then stop.
        - If the context is insufficient, state: "Insufficient evidence in the supplied context."
        """
    else:
        instructions = """
        INSTRUCTIONS:
        - Return the answer formatted in Markdown (headers, short paragraphs, numbered lists where appropriate).
        - You MAY use the recommended 7-section structure; a brief 1–2 sentence intro before the sections is allowed.
        - Do NOT mention or quote the internal context or say "based on the context".
        - If the context is insufficient, state: "Insufficient evidence in the supplied context."
        - Keep the answer concise.
        """

    # Rolling summary of the conversation so far. Raw history is also injected into
    # the messages array, but it is truncated; the summary preserves what was
    # established earlier (region, pattern, advice given) beyond that window.
    summary_section = ""
    if conversation_summary:
        summary_section = f"""
        CONVERSATION SUMMARY (what has been established earlier in this conversation; use it to stay consistent and resolve references — do not repeat it back to the user):
        {conversation_summary}
        """

    return textwrap.dedent(f"""
        CONTEXT (internal, do not describe it explicitly to the user):
        ---
        {ctx_block}
        ---
        {summary_section}
        Now answer the user's original question clearly and concisely as one integrated explanation.
        Use the retrieved context and evidence spans only as support. Do not answer a rewritten search query if it differs from the user's wording.
        {graph_instructions}
        {instructions}

        Original user question: {question}
    """).strip()

CATEGORY_LABELS = {
    "A": "Benign muscular/postural discomfort",
    "B": "MSKNeurology syndrome (TOS, scapular dyskinesis, TMJ, plexus, impingement, etc.)",
    "C": "Rare neurovascular / serious cause",
    "D": "Unclear / broad retrieval required"
}




def classify_query(user_q: str, model: str, history=None) -> str:
    """
    Agentic pre-step: classify the type of question so the RAG
    knows what retrieval domain to target.
    Uses conversation history to resolve follow-up questions.
    """
    # Build minimal history context for classification
    conv_hint = ""
    if history:
        recent = history[-4:]  # last 2 turns
        lines = []
        for turn in recent:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")[:150]
            lines.append(f"{role}: {content}")
        conv_hint = "\nRecent conversation (use to understand follow-up context):\n" + "\n".join(lines) + "\n"

    prompt = f"""
        You will classify the user's query so that a biomechanical RAG system can retrieve the correct type of sections.
{conv_hint}
        User query:
        "{user_q}"

        If this is a follow-up question (e.g. "when would I need surgery?", "what exercises?"), classify based on the TOPIC being discussed, not the question alone.

        Return ONE letter:

        A = benign muscular/postural discomfort
        B = MSKNeurology syndrome (TOS, scapular dyskinesis, TMJ, plexus, impingement, etc.)
        C = rare neurovascular/serious cause
        D = unclear / needs broad retrieval

        Return only: A, B, C, or D.
        """
    answer, _, _ = ask_openai_llm(
        prompt,
        model=model,
        num_predict=8,
        system_prompt=UTILITY_SYSTEM_PROMPT,
        temperature=UTILITY_TEMPERATURE,
        top_p=UTILITY_TOP_P,
    )
    match = re.search(r"\b([ABCD])\b", (answer or "").upper())
    return match.group(1) if match else "D"


def rewrite_query(user_q: str, category: str, openai_model: str, history=None,
                  conversation_summary: Optional[str] = None) -> str:
    """
    Rewrite the query into an MSK-biomechanics-optimized form
    based on classification category A/B/C/D.
    Uses last 2 conversation turns to resolve pronouns and vague follow-ups.
    """
    # Build recent conversation context for the rewriter. Assistant answers get a
    # larger budget: they name the pattern/region under discussion, which is exactly
    # what a vague follow-up ("what exercises help?") needs to resolve against.
    conv_context = ""
    if history:
        recent = history[-4:]  # last 2 turns (2 messages each: user + assistant)
        lines = []
        for turn in recent:
            role = turn.get("role", "user")
            max_chars = 600 if role == "assistant" else 300
            content = turn.get("content", "")[:max_chars]
            lines.append(f"{role.capitalize()}: {content}")
        conv_context = "\n".join(lines)

    history_block = ""
    if conv_context:
        history_block = f"""\nRecent conversation (use to resolve pronouns like 'it', 'that', 'this'):
{conv_context}\n"""

    # The rolling summary covers what raw history may have already truncated away:
    # the body region, the pattern under discussion, and advice already given.
    summary_block = ""
    if conversation_summary:
        summary_block = f"""\nConversation summary so far (use to resolve vague references):
{conversation_summary[:800]}\n"""

    prompt = f"""
Rewrite the user's query into a more detailed MSK biomechanics retrieval query.
{summary_block}{history_block}
Original:
"{user_q}"

Category = {category}

Rules:
- If A: emphasize benign muscular/postural mechanisms, fatigue, suboccipitals, levator, trapezius, strain patterns.
- If B: emphasize specific MSKNeurology biomechanical drivers (scapular orientation, plexus traction, rib mechanics, etc.)
- If C: emphasize neurovascular or red-flag patterns.
- If D: rewrite neutrally with maximal biomechanical detail.
- If the query references prior conversation (e.g. "what about exercises?", "tell me more"), incorporate the relevant topic from the conversation context.

Return ONLY the rewritten query, no commentary.
"""
    refined, _, _ = ask_openai_llm(
        prompt,
        model=openai_model,
        num_predict=128,
        system_prompt=UTILITY_SYSTEM_PROMPT,
        temperature=UTILITY_TEMPERATURE,
        top_p=UTILITY_TOP_P,
    )
    refined_text = (refined or "").strip().strip('"')
    if not refined_text:
        return user_q.strip()
    if len(refined_text) < max(4, min(8, len(user_q.strip()))):
        return user_q.strip()
    if not re.search(r"[A-Za-z0-9]", refined_text):
        return user_q.strip()
    return refined_text


# ── Vague-query detection ─────────────────────────────────────────────────────

_BODY_PARTS = {
    "neck", "shoulder", "back", "spine", "hip", "knee", "ankle", "foot", "feet",
    "wrist", "elbow", "hand", "finger", "thumb", "toe", "rib", "ribcage",
    "chest", "pelvis", "groin", "jaw", "tmj", "head", "skull", "clavicle",
    "scapula", "scapular", "cervical", "thoracic", "lumbar", "sacral", "sacrum",
    "coccyx", "tailbone", "trapezius", "deltoid", "bicep", "tricep", "quad",
    "hamstring", "calf", "calves", "glute", "rotator", "labrum", "meniscus",
    "disc", "disk", "facet", "si joint", "sacroiliac", "sternum", "acromion",
    "forearm", "shin", "thigh", "arm", "leg", "upper back", "lower back",
    "mid back", "scalene", "rhomboid", "levator", "pec", "lat", "oblique",
    "atlas", "axis", "c1", "c2", "c3", "c4", "c5", "c6", "c7",
    "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12",
    "l1", "l2", "l3", "l4", "l5", "s1", "s2",
}

_SYMPTOM_WORDS = {
    "numbness", "tingling", "weakness", "stiffness", "clicking", "popping",
    "grinding", "burning", "sharp", "dull", "ache", "aching", "throb",
    "throbbing", "radiating", "shooting", "stabbing", "cramping", "spasm",
    "swelling", "swollen", "tender", "sore", "soreness", "tightness", "tight",
    "limited", "restricted", "catching", "locking", "giving way", "instability",
    "pinching", "pressure", "heaviness", "fatigue", "weakness", "paresthesia",
    "dysfunction", "dyskinesis", "impingement", "radiculopathy", "neuropathy",
    "headache", "migraine", "dizziness", "vertigo", "referred",
}

_CLARIFICATION_RESPONSE = (
    "I can help, but I need one more specific description before giving an "
    "evidence-grounded answer.\n\n"
    "Please tell me the **location**, **sensation**, and **trigger/timing** in one "
    "sentence. Also mention right away if there is new or worsening weakness, "
    "bowel/bladder changes, major trauma, fever with feeling very unwell, "
    "unexplained weight loss, or severe chest pain/trouble breathing."
)

_FILLER_WORDS = {"i", "my", "me", "the", "a", "an", "is", "it", "have", "has",
                  "been", "am", "are", "was", "were", "do", "does", "did",
                  "hello", "hi", "hey", "help", "please", "thanks", "thank",
                  "some", "very", "really", "just", "also", "and", "or", "but",
                  "in", "on", "at", "to", "for", "of", "with", "from", "so",
                  "can", "could", "would", "should", "there", "here", "this",
                  "that", "what", "how", "why", "when", "where", "who",
                  "something", "feels", "feel", "wrong", "weird"}


_NEGATION_PREFIX_RE = re.compile(
    r"(?:\b(no|not|without|denies|deny|none|negative for)\b[\w\s,;/()'-]{0,45})$",
    re.I,
)

_GENERAL_INFO_RE = re.compile(
    r"^\s*(what are|what is|when should|how do|how does|explain|define|list|teach me|in general)\b",
    re.I,
)

_PERSONAL_CONCERN_RE = re.compile(
    r"\b(i|i'm|i am|i've|ive|me|my|mine|we|our|should i|can i|do i|is this|could this)\b",
    re.I,
)

_RED_FLAG_PATTERNS = {
    "bowel_bladder": [
        r"\bbowel\s+(and|or|/)\s+bladder\s+(changes?|problems?|dysfunction|incontinence|retention)\b",
        r"\bbladder\s+(and|or|/)\s+bowel\s+(changes?|problems?|dysfunction|incontinence|retention)\b",
        r"\b(bowel|bladder)\s+(changes?|problems?|dysfunction|incontinence|retention)\b",
        r"\b(trouble|difficulty|problems?)\s+(controlling|with)\s+(my\s+)?(bowel|bladder)\b",
        r"\b(lost|lose|loss of)\s+(bowel|bladder)\s+control\b",
        r"\b(can't|cannot|unable to)\s+(pee|urinate)\b",
        r"\burinary retention\b",
        r"\bsaddle\s+(anesthesia|numbness)\b",
    ],
    "progressive_neurologic_deficit": [
        r"\b(progressive|worsening|rapidly worsening|getting worse|new|sudden)\b.{0,55}\b(weakness|numbness|loss of strength|neurologic|neurological)\b",
        r"\b(weakness|numbness|loss of strength)\b.{0,55}\b(progressive|worsening|rapidly worsening|getting worse|new|sudden)\b",
        r"\b(now|suddenly|newly)\s+(can't|cannot|unable to)\s+(walk|stand|lift|raise|move|use)\b",
        r"\b(can't|cannot|unable to)\s+(walk|stand)\b",
        r"\bfoot\s+drop\b",
    ],
    "major_trauma": [
        r"\b(fall|fell|crash|crashed|accident|collision|trauma|hit)\b.{0,80}\b(severe|weakness|numbness|can't walk|cannot walk|unable to walk|deformity)\b",
        r"\b(severe|can't walk|cannot walk|unable to walk|deformity)\b.{0,50}\b(after|following)\b.{0,25}\b(fall|fell|crash|crashed|accident|collision|trauma)\b",
    ],
    "severe_chest_or_breathing": [
        r"\bsevere\s+chest\s+pain\b",
        r"\bchest\s+(pressure|tightness)\b.{0,70}\b(shortness of breath|trouble breathing|difficulty breathing|fainting|sweating|pressure)\b",
        r"\bchest\s+pain\b.{0,70}\b(shortness of breath|trouble breathing|difficulty breathing|fainting|sweating|pressure)\b",
        r"\b(shortness of breath|trouble breathing|difficulty breathing|fainting|sweating)\b.{0,70}\bchest\s+pain\b",
    ],
    "fever_systemic_decline": [
        r"\bfever\b.{0,80}\b(feel very ill|feeling very ill|systemic|chills|worsening|severe|weakness|decline)\b",
        r"\b(feel very ill|feeling very ill|systemic|chills|worsening|severe|weakness|decline)\b.{0,80}\bfever\b",
    ],
    "unexplained_weight_loss": [
        r"\b(unexplained|unintentional)\b.{0,35}\bweight\s+loss\b",
        r"\bweight\s+loss\b.{0,35}\b(unexplained|unintentional)\b",
    ],
}

_RED_FLAG_LABELS = {
    "bowel_bladder": "bowel or bladder changes",
    "progressive_neurologic_deficit": "new or worsening neurologic weakness/numbness",
    "major_trauma": "significant trauma with concerning symptoms",
    "severe_chest_or_breathing": "severe chest pain or breathing symptoms",
    "fever_systemic_decline": "fever with systemic decline or severe symptoms",
    "unexplained_weight_loss": "unexplained weight loss with concerning symptoms",
}


def _is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 70):start]
    if re.search(r"\b(not|do not|don't)\s+need\s+urgent\s+care\s+for\s+$", prefix, flags=re.I):
        return False
    return bool(_NEGATION_PREFIX_RE.search(prefix))


def _has_unnegated_match(text: str, patterns: List[str]) -> bool:
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I | re.S):
            if not _is_negated(text, match.start()):
                return True
    return False


def _looks_like_personal_or_case_description(question: str) -> bool:
    q = question.strip().lower()
    if not q:
        return False
    if _GENERAL_INFO_RE.search(q) and not _PERSONAL_CONCERN_RE.search(q):
        return False
    return True


def detect_red_flags(question: str) -> List[str]:
    """Return deterministic urgent red-flag reason codes for user/case descriptions."""
    if not _looks_like_personal_or_case_description(question):
        return []

    q = question.lower()
    reasons = []
    for reason, patterns in _RED_FLAG_PATTERNS.items():
        if _has_unnegated_match(q, patterns):
            reasons.append(reason)
    return reasons


def _red_flag_response(reasons: List[str]) -> str:
    labels = [_RED_FLAG_LABELS.get(reason, reason.replace("_", " ")) for reason in reasons]
    reason_text = ", ".join(labels) if labels else "possible red-flag symptoms"
    return (
        f"Your message mentions **{reason_text}**. I cannot determine the cause here, "
        "but these can be red flags that need **urgent in-person medical evaluation**.\n\n"
        "Please do not rely on exercises, posture changes, or a chat response to rule this out. "
        "If symptoms are severe, rapidly worsening, involve chest pain/trouble breathing, or include "
        "bowel/bladder changes, seek emergency care now. Otherwise, arrange same-day urgent medical "
        "assessment or contact a qualified clinician promptly."
    )


_MEDICATION_ADVICE_RE = re.compile(
    r"\b(should i take|can i take|dose|dosage|mg|milligram|ibuprofen|advil|naproxen|aleve|tylenol|acetaminophen|opioid|muscle relaxer|steroid|steroids|steroid injection|prescription)\b",
    re.I,
)

_DIAGNOSIS_REQUEST_RE = re.compile(
    r"\b(diagnose me|diagnose it|give me a diagnosis|exact diagnosis|diagnosis for|what diagnosis|do i have|is this definitely|confirm that|confirm (that )?i have|rule out\s+(a\s+|the\s+)?(fracture|diagnosis|condition|herniated disc|thoracic outlet|tos)|promise .* only posture)\b",
    re.I,
)

_CLEARLY_UNRELATED_RE = re.compile(
    r"\b(tax|taxes|stock|stocks|crypto|weather|recipe|cook|cooking|javascript|python code|homework|essay|dating|relationship|mortgage|car repair|car makes|when braking)\b",
    re.I,
)


def detect_scope_issue(question: str) -> Optional[str]:
    """Return a deterministic boundary issue for prompts outside product scope."""
    q = question.strip().lower()
    if not q:
        return None
    if _MEDICATION_ADVICE_RE.search(q):
        return "medication_advice"
    if _DIAGNOSIS_REQUEST_RE.search(q):
        return "diagnosis_request"
    if _CLEARLY_UNRELATED_RE.search(q):
        return "outside_msk_scope"
    return None


def _scope_boundary_response(issue: str) -> str:
    if issue == "medication_advice":
        return (
            "I can't give medication, dosage, injection, or prescription advice. This assistant is limited to "
            "educational MSK biomechanics and conservative triage. If you are considering medication or have "
            "medical conditions, pregnancy, allergies, or other medicines, ask a qualified clinician or pharmacist."
        )
    if issue == "diagnosis_request":
        return (
            "I can't diagnose you or rule conditions in or out from chat. I can help explain possible MSK "
            "biomechanical patterns, what details would matter, and when symptoms should be checked in person. "
            "If you describe the location, sensation, trigger/timing, and any urgent signs, I can give educational triage guidance."
        )
    return (
        "That is outside this assistant's scope. I can help with educational musculoskeletal biomechanics, "
        "symptom-pattern triage, conservative self-care framing, and when to seek in-person evaluation."
    )


def _static_response_payload(
    question: str,
    answer: str,
    *,
    category: str,
    category_label: str,
    triage_level: str,
    safety_gate_triggered: bool = False,
    safety_gate_reasons: Optional[List[str]] = None,
    scope_issue: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "answer": answer,
        "citations": [],
        "contexts": [],
        "retrieval_confidence": 0.0,
        "retrieval_time": 0.0,
        "generation_time": 0.0,
        "prompt_tokens": 0,
        "output_tokens": count_tokens(answer),
        "context_tokens": 0,
        "question_tokens": count_tokens(question),
        "category": category,
        "category_label": category_label,
        "original_question": question,
        "refined_query": question,
        "triage_level": triage_level,
        "safety_gate_triggered": safety_gate_triggered,
        "safety_gate_reasons": safety_gate_reasons or [],
        "scope_issue": scope_issue,
        "context_strategy": "local_preflight",
        "fallback_reason": category,
        "hierarchical_available": False,
        "selected_articles": [],
        "selected_sections": [],
        "evidence_spans": [],
        "selected_evidence_spans": [],
        "citation_map": {},
        "context_token_estimate": 0,
    }


def _low_confidence_fallback_response(question: str, retrieval_confidence: float) -> Dict[str, Any]:
    answer = (
        "I may be missing enough high-confidence source context to give a reliable mechanism-level answer yet. "
        "Could you share one sentence with the location, sensation, and trigger/timing so I can narrow this safely? "
        "If there is new or worsening weakness, bowel/bladder changes, major trauma, severe chest pain or breathing trouble, "
        "fever with feeling very unwell, or unexplained weight loss, seek urgent in-person evaluation."
    )
    return {
        "answer": answer,
        "contexts": [],
        "retrieval_time": 0.0,
        "generation_time": 0.0,
        "prompt_tokens": 0,
        "output_tokens": count_tokens(answer),
        "context_tokens": 0,
        "question_tokens": count_tokens(question),
        "citations": [],
        "retrieval_confidence": float(retrieval_confidence),
        "triage_level": "needs_more_detail",
        "original_question": question,
        "refined_query": question,
        "context_strategy": "chunk_pack",
        "fallback_reason": "low_confidence_fallback",
        "hierarchical_available": False,
        "selected_articles": [],
        "selected_sections": [],
        "evidence_spans": [],
        "selected_evidence_spans": [],
        "citation_map": {},
        "context_token_estimate": 0,
        "safety_gate_triggered": False,
        "safety_gate_reasons": [],
        "scope_issue": None,
        "low_confidence_fallback": True,
    }


def local_preflight(question: str, history=None) -> Dict[str, Any]:
    """Local, zero-cost gates that can be evaluated without model calls."""
    red_flags = detect_red_flags(question)
    if red_flags:
        answer = _red_flag_response(red_flags)
        return {
            "action": "respond",
            "kind": "red_flag_urgent",
            "result": _static_response_payload(
                question,
                answer,
                category="red_flag_urgent",
                category_label="Urgent red-flag pattern",
                triage_level="urgent_in_person_evaluation",
                safety_gate_triggered=True,
                safety_gate_reasons=red_flags,
            ),
        }

    scope_issue = detect_scope_issue(question)
    if scope_issue:
        answer = _scope_boundary_response(scope_issue)
        return {
            "action": "respond",
            "kind": "scope_boundary",
            "result": _static_response_payload(
                question,
                answer,
                category="scope_boundary",
                category_label="Outside product scope",
                triage_level="scope_boundary",
                scope_issue=scope_issue,
            ),
        }

    if not history and not _GENERAL_INFO_RE.search(question.strip().lower()) and _is_vague_query(question):
        return {
            "action": "respond",
            "kind": "clarification",
            "result": _static_response_payload(
                question,
                _CLARIFICATION_RESPONSE,
                category="clarification",
                category_label="Needs more detail",
                triage_level="needs_more_detail",
            ),
        }

    return {"action": "continue", "kind": "continue", "result": None}


def _is_vague_query(question: str) -> bool:
    """Return True if the question lacks enough anatomical/symptom specificity."""
    q_lower = question.lower()
    words = set(re.findall(r"[a-z0-9]+", q_lower))

    # Check for body parts (including multi-word like "lower back", "si joint")
    has_body_part = any(bp in q_lower for bp in _BODY_PARTS)

    # Check for symptom descriptors
    has_symptom = bool(words & _SYMPTOM_WORDS)

    # If both are missing and the query has few meaningful words → vague
    meaningful_words = words - _FILLER_WORDS
    # "pain" alone doesn't count as enough specificity
    meaningful_no_pain = meaningful_words - {"pain", "hurt", "hurts", "hurting", "problem", "issue", "wrong"}

    if len(meaningful_no_pain) <= 1:
        return True

    if has_body_part and has_symptom:
        return False  # Clearly specific
    if has_body_part and len(meaningful_words) >= 3:
        return False  # Body part + some context
    if has_symptom and len(meaningful_words) >= 4:
        return False  # Symptom + enough context

    # If there's at least one body part and >5 meaningful words, it's probably fine
    if has_body_part and len(meaningful_no_pain) >= 3:
        return False

    # Everything else with fewer than 5 meaningful words is too vague
    if len(meaningful_no_pain) < 3:
        return True

    return False


def agentic_run(
    question: str,
    cfg: Optional[QAConfig] = None,
    history=None,
    on_token = None,
    conversation_summary: Optional[str] = None,
):
    """Public entry point. Binds any per-request BYO OpenAI key for the duration of
    the request (inside this thread) and always unbinds it afterwards, so a
    user-supplied key is never cached process-wide or leaked into another request."""
    cfg = cfg or QAConfig()
    key_token = _request_api_key.set(cfg.api_key or None)
    try:
        return _agentic_run_impl(
            question, cfg, history=history, on_token=on_token,
            conversation_summary=conversation_summary,
        )
    finally:
        _request_api_key.reset(key_token)


def _agentic_run_impl(
    question: str,
    cfg: Optional[QAConfig] = None,
    history=None,
    on_token = None,
    conversation_summary: Optional[str] = None,
):
    cfg = cfg or QAConfig()

    # The rolling summary is client-supplied; cap it defensively before use.
    conversation_summary = (conversation_summary or "").strip()[:2500] or None

    # Step -1: local zero-cost gates before retrieval/generation.
    preflight = local_preflight(question, history=history)
    if preflight["action"] == "respond":
        result = preflight["result"]
        # Static responses (red flag / scope / clarification) are not folded into
        # the rolling summary; the client keeps what it had.
        result["conversation_summary"] = conversation_summary
        if on_token:
            on_token(result["answer"])
        return result

    # Steps 1 & 2 (classify + rewrite) are best-effort LLM pre-processing. If the key
    # is dead / out of quota (or the utility model fails for any reason), degrade to a
    # broad category and the original query so retrieval can still run instead of
    # crashing the whole request before it reaches the retrieval/generation fallbacks.
    query_processing_degraded = False

    # Step 1: classify
    try:
        category = classify_query(question, RERANKER_MODEL, history=history)
    except Exception as exc:
        query_processing_degraded = True
        category = "D"  # broad retrieval
        logger.warning("classify_query degraded to 'D' (%s: %s)", type(exc).__name__, exc)

    # Step 2: rewrite for retrieval (with history for context-aware rewriting)
    try:
        refined_q = rewrite_query(question, category, RERANKER_MODEL, history=history,
                                  conversation_summary=conversation_summary)
    except Exception as exc:
        query_processing_degraded = True
        refined_q = question  # fall back to the original user question
        logger.warning("rewrite_query degraded to original question (%s: %s)", type(exc).__name__, exc)

    generation_question = question if cfg.answer_original_question else refined_q

    # Step 3: run run_qa() but forward history correctly. The rewritten query is
    # used for retrieval; generation answers the original user question by default.
    if history:
        res = run_qa(refined_q, config=cfg, on_token=on_token, history=history,
                     generation_question=generation_question,
                     conversation_summary=conversation_summary)
    else:
        res = run_qa(refined_q, config=cfg, on_token=on_token,
                     generation_question=generation_question,
                     conversation_summary=conversation_summary)

    # Step 4: fold this exchange into the rolling summary for the next turn.
    # Best-effort: only after a real LLM answer (if generation degraded, the utility
    # model is likely down too), and never fatal — the previous summary is kept.
    new_summary = conversation_summary
    if str(res.get("answer_mode", "")).startswith("llm:") and res.get("answer"):
        try:
            new_summary = summarize_conversation(conversation_summary, question, res["answer"])
        except Exception as exc:
            logger.warning("summarize_conversation failed (%s: %s); keeping previous summary",
                           type(exc).__name__, exc)
    res["conversation_summary"] = new_summary

    res["category"] = category
    res["category_label"] = CATEGORY_LABELS.get(category, "Unknown")
    res["refined_query"] = refined_q
    res["original_question"] = question
    res["query_processing_degraded"] = query_processing_degraded
    res.setdefault("triage_level", "educational_triage")
    res.setdefault("safety_gate_triggered", False)
    res.setdefault("safety_gate_reasons", [])
    return res



def summarize_conversation(prev_summary: Optional[str], user_q: str, answer: str,
                           model: str = RERANKER_MODEL) -> Optional[str]:
    """Fold the latest exchange into a compact rolling conversation summary.

    The summary is carried by the client between turns (the backend is stateless),
    injected into the rewriter and the generation prompt, and lets the conversation
    stay coherent beyond the raw-history window. Uses the cheap utility model; the
    caller is expected to treat failures as non-fatal and keep the previous summary.
    """
    prev = (prev_summary or "").strip()[:1500]
    q = (user_q or "").strip()[:500]
    a = (answer or "").strip()[:1500]
    if not q or not a:
        return prev_summary

    prev_block = prev if prev else "(none — this is the first exchange)"
    prompt = f"""
Maintain a rolling summary of an MSK biomechanics triage conversation.

Current summary:
{prev_block}

Latest exchange:
User: {q}
Assistant: {a}

Update the summary to include the latest exchange. Keep it under 120 words. Capture, when present:
- body region(s) and symptoms the user described
- the biomechanical pattern or mechanism identified
- advice or corrections already given
- anything the user was asked to clarify or is still deciding

Write plain prose (no headers, no bullets). Do not add information that is not in the summary or the exchange. Return ONLY the summary text.
""".strip()

    summary, _, _ = ask_openai_llm(
        prompt,
        model=model,
        num_predict=220,
        system_prompt=UTILITY_SYSTEM_PROMPT,
        temperature=UTILITY_TEMPERATURE,
        top_p=UTILITY_TOP_P,
    )
    summary = " ".join((summary or "").split()).strip()
    return summary[:2000] or prev_summary


def _truncate_history(history, max_turns=5, max_chars_user=800,
                      max_chars_assistant=1600, max_total_tokens=3000):
    """
    Prepare conversation history for the LLM messages array.
    - Keeps the last `max_turns` pairs (10 messages max)
    - Truncates each message by role: assistant answers carry the established
      clinical context of the conversation, so they get a larger character
      budget than user messages
    - Fills the token budget newest-first, so when the budget runs out it is
      the oldest messages that are dropped
    """
    if not history:
        return []

    # Take the last N messages (max_turns * 2 for user+assistant pairs)
    recent = history[-(max_turns * 2):]

    truncated = []
    total_tokens = 0

    for msg in reversed(recent):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Truncate long messages
        max_chars = max_chars_assistant if role == "assistant" else max_chars_user
        if len(content) > max_chars:
            content = content[:max_chars] + "…"

        est_tokens = len(content) // 4  # rough estimate: 1 token ≈ 4 chars
        if total_tokens + est_tokens > max_total_tokens:
            break

        total_tokens += est_tokens
        truncated.append({"role": role, "content": content})

    truncated.reverse()  # restore chronological order
    return truncated


def build_evidence_only_answer(
    context: List[Dict[str, Any]],
    context_pack: Optional[ContextPack],
    question: str,
    max_items: int = 4,
) -> str:
    """Deterministic, grounded answer assembled directly from retrieved evidence when
    no generation model is available.

    Conservative by construction: it surfaces what the corpus actually says, adds no
    diagnosis and no reassurance, and keeps the standard educational / seek-in-person
    framing. Red-flag prompts never reach this path — they short-circuit earlier in
    local_preflight — so this is only used for educational MSK queries.
    """
    lines: List[str] = [
        "_The AI answer-writer is temporarily unavailable, so this reply is assembled "
        "directly from the most relevant passages retrieved from the MSK knowledge base "
        "— no interpretation added._",
        "",
    ]

    spans = list(getattr(context_pack, "selected_evidence_spans", None) or [])
    items: List[Tuple[str, str, str]] = []
    if spans:
        for sp in spans[:max_items]:
            src = sp.get("source_relpath") or sp.get("title") or "source"
            section = sp.get("section_name") or sp.get("title") or ""
            items.append((str(src), str(section), (sp.get("text") or "").strip()))
    else:
        for it in context[:max_items]:
            meta = it.get("meta", {}) or {}
            src = meta.get("source_relpath") or meta.get("title") or "source"
            section = meta.get("section") or ""
            items.append((str(src), str(section), (it.get("text") or "").strip()))

    if not items:
        lines.append("No supporting passages were retrieved for this question.")
    else:
        lines.append("**Most relevant evidence from the knowledge base:**")
        lines.append("")
        for i, (src, section, text) in enumerate(items, start=1):
            snippet = _truncate_text_tokens(text, 160) if text else ""
            lines.append(f"**{i}. {section}**" if section else f"**{i}.**")
            if snippet:
                lines.append(f"> {snippet}")
            lines.append(f"— source: {src}")
            lines.append("")

    lines.append(
        "This is educational information about musculoskeletal biomechanics, not a "
        "diagnosis. For persistent, worsening, or concerning symptoms — or any red-flag "
        "signs such as new weakness or numbness, bowel or bladder changes, fever with "
        "feeling unwell, or unexplained weight loss — seek in-person evaluation from a "
        "qualified clinician."
    )
    return "\n".join(lines).strip()


def ask_openai_llm(
    prompt: str,
    model: str,
    num_predict: int,
    on_token=None,
    history=None,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    client: Optional[OpenAI] = None,
    supports_streaming: bool = True,
    param_style: str = "openai",
):
    """
    Clean, stable Chat Completions wrapper.

    Works for the native OpenAI API and for OpenAI-compatible fallback providers
    (Groq, Cerebras, OpenRouter, Mistral, Gemini) via an injected `client`. Providers
    differ in small but real ways, controlled by `param_style`:
      - "openai": native API — uses `max_completion_tokens` and, for gpt-5, pins
        `reasoning_effort=none`.
      - "compat": generic OpenAI-compatible endpoint — uses `max_tokens` and passes
        temperature/top_p through when provided.
    `supports_streaming=False` skips the streaming attempt entirely.
    """

    if client is None:
        client = _get_openai_client()

    # Token counting for telemetry
    prompt_tokens = count_tokens(prompt)

    # Build multi-turn messages array
    messages = [{"role": "system", "content": system_prompt}]

    # Inject conversation history for multi-turn context
    conv_history = _truncate_history(history)
    if conv_history:
        messages.extend(conv_history)

    messages.append({"role": "user", "content": prompt})

    parts = []
    if param_style == "openai":
        request_args = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": num_predict,
        }
        if model.startswith("gpt-5"):
            # GPT-5 family: pin reasoning effort so the output-token budget goes to
            # the answer, and skip sampling params the API rejects.
            request_args["reasoning_effort"] = "none"
        else:
            if temperature is not None:
                request_args["temperature"] = temperature
            if top_p is not None:
                request_args["top_p"] = top_p
    else:  # "compat" — most OpenAI-compatible providers expect max_tokens
        request_args = {
            "model": model,
            "messages": messages,
            "max_tokens": num_predict,
        }
        if temperature is not None:
            request_args["temperature"] = temperature
        if top_p is not None:
            request_args["top_p"] = top_p

    # ---------- 1) Try streaming first (if the provider supports it) ----------
    if supports_streaming:
        try:
            stream = client.chat.completions.create(**request_args, stream=True)

            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if not delta:
                    continue

                text = getattr(delta, "content", None)
                if text:
                    parts.append(text)
                    if on_token:
                        on_token(text)

            answer = "".join(parts)
            output_tokens = count_tokens(answer)
            return answer, int(prompt_tokens), int(output_tokens)

        except Exception as stream_err:
            # A key/quota/auth failure is terminal — do not retry non-streaming (it
            # would just fail again) and surface the typed error so callers can fall back.
            code = _classify_openai_error(stream_err)
            if code:
                raise OpenAIKeyError(f"Generation request failed: {stream_err}", code=code) from stream_err
            # Only fall back for streaming-specific issues, not API errors
            err_str = str(stream_err)
            if "invalid" in err_str.lower() or "400" in err_str:
                raise RuntimeError(f"OpenAI API error: {stream_err}")
            # Fall back to non-streaming for other issues


    # If streaming already emitted partial tokens but then failed, do NOT retry
    # non-streaming — that would duplicate output on the client.
    if parts:
        raise RuntimeError("Streaming failed after partial output; not retrying non-streaming.")

    # ---------- 2) Non-streaming fallback ----------
    try:
        resp = client.chat.completions.create(**request_args, stream=False)
        content = resp.choices[0].message.content or ""
        answer = content.strip()
        output_tokens = count_tokens(answer)
        if on_token and answer:
            on_token(answer)
        return answer, int(prompt_tokens), int(output_tokens)

    except Exception as e:
        code = _classify_openai_error(e)
        if code:
            raise OpenAIKeyError(f"Generation request failed: {e}", code=code) from e
        raise RuntimeError(f"Chat completions failed: {e}")


# ── Free fallback generation providers ───────────────────────────────────────
#
# Optional OpenAI-compatible providers tried (in order) when the primary OpenAI
# generation call is unavailable. Each is configured purely via environment
# variables and is skipped unless its API key is set. Streaming and parameter
# behavior differ per provider, so each has an explicit adapter spec rather than
# assuming a single uniform contract.

@dataclasses.dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    api_key: str
    model: str
    extra_headers: Optional[Dict[str, str]] = None
    supports_streaming: bool = True
    param_style: str = "compat"


_DEFAULT_PROVIDER_ORDER = ["groq", "cerebras", "openrouter", "mistral", "gemini"]

# name -> static adapter definition (base_url + which env vars carry key/model +
# a sensible default model + any per-provider quirks).
_PROVIDER_DEFS: Dict[str, Dict[str, Any]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        # llama-3.3-70b-versatile / llama-3.1-8b-instant were deprecated by Groq on
        # 2026-06-17 and shut down 2026-08-16. gpt-oss-120b is Groq's recommended
        # replacement for the 70B and a current production model. Override via GROQ_MODEL.
        "default_model": "openai/gpt-oss-120b",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "model_env": "CEREBRAS_MODEL",
        "default_model": "llama-3.3-70b",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_model": "openrouter/auto",
        "extra_headers": {"X-Title": "MSK Triage Chatbot"},
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "model_env": "MISTRAL_MODEL",
        "default_model": "mistral-small-latest",
    },
    "gemini": {
        # Google's OpenAI-compatibility layer.
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-2.0-flash",
    },
}


def _configured_providers() -> List[ProviderSpec]:
    """Return the fallback providers whose API keys are set, in priority order.

    Order comes from the FALLBACK_PROVIDERS env (comma-separated) if set, else the
    default order. Providers with no configured key are skipped.
    """
    raw_order = [p.strip() for p in os.getenv("FALLBACK_PROVIDERS", "").split(",") if p.strip()]
    order = raw_order or _DEFAULT_PROVIDER_ORDER

    specs: List[ProviderSpec] = []
    for name in order:
        definition = _PROVIDER_DEFS.get(name)
        if not definition:
            continue
        api_key = os.getenv(definition["api_key_env"])
        if not api_key:
            continue
        specs.append(ProviderSpec(
            name=name,
            base_url=definition["base_url"],
            api_key=api_key,
            model=os.getenv(definition["model_env"]) or definition["default_model"],
            extra_headers=definition.get("extra_headers"),
            supports_streaming=definition.get("supports_streaming", True),
            param_style=definition.get("param_style", "compat"),
        ))
    return specs


# Curated model suggestions per provider for the UI dropdown. These are convenient
# presets only — a custom model string is also accepted per provider.
_SUGGESTED_MODELS: Dict[str, List[str]] = {
    "openai": ["gpt-5.4-mini", "gpt-4.1-mini", "gpt-4.1-nano"],
    # Groq's llama-3.3-70b-versatile and llama-3.1-8b-instant shut down 2026-08-16;
    # replaced here by Groq's current production gpt-oss models (the vendor-
    # recommended migration). Any served model id can still be typed (allow_custom).
    "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    "cerebras": ["llama-3.3-70b", "llama3.1-8b"],
    "openrouter": ["openrouter/auto"],
    "mistral": ["mistral-small-latest", "mistral-large-latest"],
    "gemini": ["gemini-2.0-flash", "gemini-2.0-flash-lite"],
}

# Providers that require the user's own key (the server key may be dead/absent).
_PREMIUM_PROVIDERS = {"openai"}


def _provider_spec_for(name: str, model_override: Optional[str] = None) -> Optional[ProviderSpec]:
    """Build a ProviderSpec for a single free provider by name, or None if its
    server-side API key is not configured. Used to pin generation to a chosen
    provider regardless of its position in the fallback priority order."""
    definition = _PROVIDER_DEFS.get(name)
    if not definition:
        return None
    api_key = os.getenv(definition["api_key_env"])
    if not api_key:
        return None
    return ProviderSpec(
        name=name,
        base_url=definition["base_url"],
        api_key=api_key,
        model=model_override or os.getenv(definition["model_env"]) or definition["default_model"],
        extra_headers=definition.get("extra_headers"),
        supports_streaming=definition.get("supports_streaming", True),
        param_style=definition.get("param_style", "compat"),
    )


def generation_catalog() -> Dict[str, Any]:
    """Describe the generation providers/models the UI may offer.

    Free providers appear as selectable only when their server-side key is configured
    (``server_key``); the premium provider (OpenAI) is always listed but flagged
    ``requires_user_key`` since the server key may be exhausted. Model lists are
    suggestions — a custom model string is also accepted per provider. The default is
    the first configured free provider, else OpenAI.
    """
    configured = {spec.name for spec in _configured_providers()}
    providers: List[Dict[str, Any]] = [{
        "name": "openai",
        "label": "OpenAI",
        "tier": "premium",
        "server_key": bool(os.getenv("OPENAI_API_KEY")),
        "requires_user_key": True,
        "default_model": OPENAI_MODEL,
        "models": list(_SUGGESTED_MODELS["openai"]),
        "allow_custom": True,
    }]
    for name in _DEFAULT_PROVIDER_ORDER:
        definition = _PROVIDER_DEFS.get(name)
        if not definition:
            continue
        providers.append({
            "name": name,
            "label": name.capitalize(),
            "tier": "free",
            "server_key": name in configured,
            "requires_user_key": False,
            "default_model": os.getenv(definition["model_env"]) or definition["default_model"],
            "models": list(_SUGGESTED_MODELS.get(name, [definition["default_model"]])),
            "allow_custom": True,
        })
    default_provider = next(
        (p["name"] for p in providers if p["tier"] == "free" and p["server_key"]),
        "openai",
    )
    default_model = next(
        (p["default_model"] for p in providers if p["name"] == default_provider),
        OPENAI_MODEL,
    )
    return {
        "providers": providers,
        "default_provider": default_provider,
        "default_model": default_model,
    }


def _call_provider(
    spec: ProviderSpec,
    prompt: str,
    num_predict: int,
    on_token=None,
    history=None,
    system_prompt: str = SYSTEM_PROMPT,
) -> Tuple[str, int, int]:
    """Call a single OpenAI-compatible fallback provider via its adapter spec."""
    if OpenAI is None:
        raise RuntimeError("The openai package is required for provider calls.")
    client = OpenAI(
        api_key=spec.api_key,
        base_url=spec.base_url,
        default_headers=spec.extra_headers or None,
    )
    return ask_openai_llm(
        prompt,
        model=spec.model,
        num_predict=num_predict,
        on_token=on_token,
        history=history,
        system_prompt=system_prompt,
        client=client,
        supports_streaming=spec.supports_streaming,
        param_style=spec.param_style,
    )


def generate_answer_with_fallback(
    prompt: str,
    cfg: "QAConfig",
    context: List[Dict[str, Any]],
    context_pack: Optional[ContextPack],
    question: str,
    on_token=None,
    history=None,
) -> Tuple[str, int, int, str, Optional[str]]:
    """Generate an answer, degrading gracefully.

    Two modes:
      * **Pinned** (``cfg.generation_provider`` set) — the user picked a specific
        provider+model in the UI. Use exactly that; on failure go straight to the
        deterministic evidence-only answer (predictable: you get what you picked or a
        safe grounded answer, never a silently different model).
      * **Default** (no pin) — the resilience chain:
        OpenAI (effective key) -> free providers (Groq, Cerebras, ...) -> evidence-only.

    Returns (answer, prompt_tokens, output_tokens, answer_mode, generation_model). The
    last element is the model that actually produced the answer (None for evidence-only).
    To keep streamed output coherent, a provider is only abandoned for the next one if it
    failed *before* emitting any tokens; a mid-stream failure is surfaced to the caller.
    """
    tokens_emitted = 0

    def _tracked(tok: str):
        nonlocal tokens_emitted
        tokens_emitted += 1
        if on_token:
            on_token(tok)

    def _empty(text: str) -> bool:
        # A provider can return HTTP 200 with empty/whitespace content (e.g. a
        # reasoning model that spends its whole token budget before emitting an
        # answer). That streams zero tokens and must NOT be treated as a success —
        # otherwise the client shows a dead "No response received" bubble. Only
        # meaningful when nothing was streamed yet; a mid-stream truncation still
        # produced real output for the user.
        return tokens_emitted == 0 and not (text or "").strip()

    def _evidence_only() -> Tuple[str, int, int, str, Optional[str]]:
        text = build_evidence_only_answer(context, context_pack, question)
        if on_token and tokens_emitted == 0:
            on_token(text)
        return text, count_tokens(prompt), count_tokens(text), "evidence_only", None

    # ── Pinned selection: honor the user's explicit provider+model choice. ──
    pinned = (cfg.generation_provider or "").strip().lower()
    pinned_model = (cfg.generation_model or "").strip() or None
    if pinned:
        if pinned == "openai":
            model = pinned_model or cfg.openai_model
            try:
                text, pt, ot = ask_openai_llm(
                    prompt, model=model, num_predict=cfg.num_predict,
                    on_token=_tracked, history=history,
                )
                if _empty(text):
                    logger.warning("pinned openai model %s returned empty output; using evidence-only", model)
                else:
                    return text, pt, ot, "llm:openai", model
            except OpenAIKeyError as exc:
                if tokens_emitted > 0:
                    raise
                logger.warning("pinned openai model unavailable (%s); using evidence-only", exc)
        else:
            spec = _provider_spec_for(pinned, pinned_model)
            if spec is None:
                logger.warning("pinned provider %s has no server key; using evidence-only", pinned)
            else:
                try:
                    text, pt, ot = _call_provider(
                        spec, prompt, cfg.num_predict, on_token=_tracked, history=history,
                    )
                    if _empty(text):
                        logger.warning("pinned provider %s returned empty output; using evidence-only", pinned)
                    else:
                        return text, pt, ot, f"llm:{spec.name}", spec.model
                except Exception as exc:
                    if tokens_emitted > 0:
                        raise
                    logger.warning(
                        "pinned provider %s failed (%s: %s); using evidence-only",
                        pinned, type(exc).__name__, exc,
                    )
        return _evidence_only()

    # ── Default resilience chain (no explicit pin). ──
    # 1) Primary: OpenAI (uses the per-request key if supplied, else the env key).
    try:
        text, pt, ot = ask_openai_llm(
            prompt, model=cfg.openai_model, num_predict=cfg.num_predict,
            on_token=_tracked, history=history,
        )
        if not _empty(text):
            return text, pt, ot, "llm:openai", cfg.openai_model
        logger.warning("openai generation returned empty output; trying fallback providers")
    except OpenAIKeyError as exc:
        if tokens_emitted > 0:
            raise
        logger.warning("openai generation unavailable (%s); trying fallback providers", exc)

    # 2) Free OpenAI-compatible providers, in priority order.
    for spec in _configured_providers():
        try:
            text, pt, ot = _call_provider(
                spec, prompt, cfg.num_predict, on_token=_tracked, history=history,
            )
            if _empty(text):
                logger.warning("fallback provider %s returned empty output; trying next", spec.name)
                continue
            return text, pt, ot, f"llm:{spec.name}", spec.model
        except Exception as exc:
            if tokens_emitted > 0:
                raise
            logger.warning(
                "fallback provider %s failed (%s: %s); trying next",
                spec.name, type(exc).__name__, exc,
            )
            continue

    # 3) Last resort: deterministic evidence-only answer (never fails).
    logger.warning("all generation providers unavailable; using evidence-only answer")
    return _evidence_only()








def run_qa(
    question: str,
    config: Optional[QAConfig] = None,
    *,
    on_token: Optional[Callable[[str], None]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    generation_question: Optional[str] = None,
    conversation_summary: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = config or QAConfig()
    answer_question = generation_question or question
    first_token_latency = None
    generation_model: Optional[str] = None

    effective_budget_tokens = (
        cfg.budget_tokens if cfg.budget_tokens and cfg.budget_tokens > 0
        else words_to_tokens_heuristic(cfg.budget_words)
    )

    coll = _backend.load_collection()

    t0 = time.time()

    # Reset per-request degradation flag; hybrid_search sets it if dense retrieval fails.
    _retrieval_degraded.set(False)

    # ---- Hybrid search (dense + BM25) with multi-query ----
    raw = hybrid_search(question, coll, retrieval_pool=cfg.retrieval_pool)

    # Multi-query expansion is only used when initial confidence is weak — and never
    # when degraded to BM25-only, since it needs both the utility LLM and embeddings.
    initial_confidence = raw_top_confidence(raw)
    if not _retrieval_degraded.get() and initial_confidence < MULTI_QUERY_TRIGGER_CONFIDENCE:
        alt_queries = generate_multi_queries(question, n=MULTI_QUERY_COUNT)
        alt_pool = max(1, int(cfg.retrieval_pool * MULTI_QUERY_RETRIEVAL_RATIO))
        alt_raws = [raw]
        for alt_q in alt_queries:
            alt_raws.append(hybrid_search(alt_q, coll, retrieval_pool=alt_pool))
        raw = merge_raw_results(*alt_raws)

    retrieval_mode = "bm25_only" if _retrieval_degraded.get() else "hybrid"
    retrieval_time = time.time() - t0

    if not raw or not raw.get("documents") or not raw["documents"][0]:
        return {
            "answer": "No results found in the corpus.",
            "contexts": [],
            "retrieval_time": retrieval_time,
            "generation_time": 0.0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "context_tokens": 0,
            "question_tokens": count_tokens(question),
            "citations": [],
            "retrieval_confidence": 0.0,  # TIER1
            "retrieval_mode": retrieval_mode,
            "answer_mode": "no_context",
            "generation_model": None,
            "original_question": answer_question,
            "refined_query": question,
            "context_strategy": "chunk_pack",
            "fallback_reason": "no_results",
            "hierarchical_available": False,
            "selected_articles": [],
            "selected_sections": [],
            "evidence_spans": [],
            "selected_evidence_spans": [],
            "citation_map": {},
            "context_token_estimate": 0,
        }

    # ---- compute retrieval confidence (TIER1) ----
    corpus_confidence = raw_top_confidence(raw)
    retrieval_confidence = corpus_confidence  # TIER1

    if cfg.enable_low_confidence_fallback and retrieval_confidence < cfg.low_confidence_fallback_threshold:
        out = _low_confidence_fallback_response(answer_question, retrieval_confidence)
        out["retrieval_time"] = retrieval_time
        out["original_question"] = answer_question
        out["refined_query"] = question
        out["context_strategy"] = "chunk_pack"
        out["fallback_reason"] = "low_confidence_fallback"
        out["hierarchical_available"] = False
        return out

    # ---- biases ----
    if cfg.use_bias:
        biased = apply_bias(question, raw)
    else:
        # Build unbiased items directly from raw distances
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        dists = raw["distances"][0]
        biased = [{"text": d, "meta": m, "dist": float(dist)}
                  for d, m, dist in zip(docs, metas, dists)]

    # ---- history gating ----
    memory_docs: List[Dict[str, Any]] = []
    if cfg.include_history and history:
        if corpus_confidence < cfg.history_use_threshold:
            memory_docs = select_relevant_history(
                history,
                question,
                max_turns=cfg.history_max_turns,
                max_entries=cfg.history_top_entries,
                decay_factor=cfg.history_decay,
                scale=cfg.history_scale,
                dist_penalty=cfg.history_dist_penalty,
            )

    merged_candidates = memory_docs + biased
    grouped = group_by_source(merged_candidates)

    # limit per-source pool
    for src in list(grouped.keys()):
        grouped[src] = sorted(grouped[src], key=lambda x: x["dist"])[:cfg.per_source_pool]

    # ---- RERANK ----
    if cfg.use_reranker:
        print(f"[RERANKER] model={RERANKER_MODEL} | top_n={cfg.reranker_top_n} | use={cfg.use_reranker}")
        for src, group in list(grouped.items()):
            grouped[src] = maybe_rerank(
                question,
                group,
                cfg.reranker_top_n,
            )



    # ---- TIER1 FAILSAFE ----
    flat_after = sum(grouped.values(), [])
    if len(flat_after) == 0:
        grouped = group_by_source(biased)

    # flatten
    candidates: List[Dict[str, Any]] = []
    for src, group in grouped.items():
        candidates.extend(sorted(group, key=lambda x: x["dist"]))

    candidates = candidates[:cfg.final_limit]

    context = pick_multichunk_context(
        items=candidates,
        top_k=cfg.top_k,
        per_source_max=cfg.per_source_max,
        budget_tokens=effective_budget_tokens,
        neighbor_headroom=cfg.neighbor_headroom,
    )

    # ---- Context compression: keep only the most relevant sentences ----
    context = compress_context(context, question)

    if not context:
        return {
            "answer": "No usable context under the current token budget.",
            "contexts": [],
            "retrieval_time": retrieval_time,
            "generation_time": 0.0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "context_tokens": 0,
            "question_tokens": count_tokens(question),
            "citations": [],
            "retrieval_confidence": float(retrieval_confidence),  # TIER1
            "retrieval_mode": retrieval_mode,
            "answer_mode": "no_context",
            "generation_model": None,
            "original_question": answer_question,
            "refined_query": question,
            "context_strategy": "chunk_pack",
            "fallback_reason": "no_usable_context",
            "hierarchical_available": False,
            "selected_articles": [],
            "selected_sections": [],
            "evidence_spans": [],
            "selected_evidence_spans": [],
            "citation_map": {},
            "context_token_estimate": 0,
        }

    context_pack = build_context_pack(context, question, cfg)

    # prompt + generation
    prompt = build_prompt(answer_question, context, history=history, context_pack=context_pack,
                          conversation_summary=conversation_summary)
    context_tokens = context_pack.token_estimate
    question_tokens = count_tokens(answer_question)

    if cfg.generate_answer:
        t1 = time.time()
        first_token_time = None

        def token_callback(tok):
            nonlocal first_token_time
            if first_token_time is None:
                first_token_time = time.time()
            if on_token:
                on_token(tok)

        # Generate with graceful degradation: OpenAI -> free providers -> evidence-only.
        answer_text, prompt_tokens, output_tokens, answer_mode, generation_model = generate_answer_with_fallback(
            prompt,
            cfg,
            context,
            context_pack,
            answer_question,
            on_token=token_callback,
            history=history,
        )

        gen_time = time.time() - t1

        # Attach timing metric for Streamlit telemetry
        if first_token_time is not None:
            first_token_latency = first_token_time - t1
        else:
            first_token_latency = None

    else:
        answer_text = ""
        prompt_tokens = 0
        output_tokens = 0
        gen_time = 0.0
        answer_mode = "disabled"

    uniq: List[str] = []
    for it in context:
        src = it["meta"].get("source_relpath", "unknown")
        sec = it["meta"].get("section", "n/a")
        entry = f"{src} — {sec}"
        if entry not in uniq:
            uniq.append(entry)

    return {
        "answer": (answer_text or "").strip(),
        "contexts": context,
        "original_question": answer_question,
        "refined_query": question,
        "context_strategy": context_pack.strategy,
        "fallback_reason": context_pack.fallback_reason,
        "hierarchical_available": context_pack.hierarchical_available,
        "selected_articles": context_pack.selected_articles,
        "selected_sections": context_pack.selected_sections,
        "evidence_spans": context_pack.selected_evidence_spans,
        "selected_evidence_spans": context_pack.selected_evidence_spans,
        "citation_map": context_pack.citation_map,
        "context_token_estimate": context_pack.token_estimate,
        "total_context_token_estimate": context_pack.total_context_token_estimate or context_pack.token_estimate,
        "graph_available": context_pack.graph_available,
        "graph_fallback_reason": context_pack.graph_fallback_reason,
        "graph_nodes": context_pack.graph_nodes,
        "graph_edges": context_pack.graph_edges,
        "graph_paths": context_pack.graph_paths,
        "graph_supporting_spans": context_pack.graph_supporting_spans,
        "graph_context_token_estimate": context_pack.graph_context_token_estimate,
        "graph_context_strategy": context_pack.graph_context_strategy,
        "graph_focus_context": context_pack.graph_focus_context,
        "graph_context_focused": context_pack.graph_context_focused,
        "retrieval_time": retrieval_time,
        "generation_time": gen_time,
        "first_token_latency": first_token_latency,
        "prompt_tokens": prompt_tokens,      # computed manually
        "output_tokens": output_tokens,      # computed manually
        "context_tokens": context_tokens,
        "question_tokens": question_tokens,
        "citations": uniq,
        "retrieval_confidence": float(retrieval_confidence),
        "retrieval_mode": retrieval_mode,
        "answer_mode": answer_mode,
        "generation_model": generation_model,
    }



def log_interaction(
    question: str,
    result: Dict[str, Any],
    log_dir: Path = PROJECT_ROOT / "logs" / "sessions",
) -> None:
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.time(),
            "q": question,
            "answer": result.get("answer"),
            "retrieval_time": result.get("retrieval_time"),
            "generation_time": result.get("generation_time"),
            "prompt_tokens": result.get("prompt_tokens"),
            "output_tokens": result.get("output_tokens"),
            "context_tokens": result.get("context_tokens"),
            "question_tokens": result.get("question_tokens"),
            "citations": result.get("citations"),
            "contexts_meta": [
                {"src": it["meta"].get("source_relpath"), "sec": it["meta"].get("section")}
                for it in result.get("contexts", [])
            ],
        }
        out = log_dir / f"session_{int(time.time())}.jsonl"
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser(description="MSK Neurology RAG (v7.6)")
    p.add_argument("--q", type=str, help="One-off question")

    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--per-source-max", type=int, default=PER_SOURCE_MAX_CHUNKS)
    p.add_argument("--retrieval-pool", type=int, default=RETRIEVAL_POOL)
    p.add_argument("--per-source-pool", type=int, default=PER_SOURCE_POOL)
    p.add_argument("--final-limit", type=int, default=FINAL_LIMIT)

    p.add_argument("--budget-tokens", type=int, default=BUDGET_TOKENS)
    p.add_argument("--budget-words", type=int, default=BUDGET_WORDS_DEPRECATED)
    p.add_argument("--neighbor-headroom", type=int, default=NEIGHBOR_HEADROOM)

    p.add_argument("--num-predict", type=int, default=NUM_PREDICT)
    p.add_argument("--openai-model", type=str, default=OPENAI_MODEL)
    

    p.add_argument("--use-reranker", action="store_true")
    
    p.add_argument("--reranker-top-n", type=int, default=10)

    p.add_argument("--include-history", action="store_true")
    p.add_argument("--history-max-turns", type=int, default=10)
    p.add_argument("--history-top-entries", type=int, default=DEFAULT_HISTORY_TOP_ENTRIES)
    p.add_argument("--history-decay", type=float, default=DEFAULT_HISTORY_DECAY)
    p.add_argument("--history-scale", type=float, default=DEFAULT_HISTORY_SCALE)
    p.add_argument("--history-dist-penalty", type=float, default=DEFAULT_HISTORY_DIST_PENALTY)
    p.add_argument("--history-use-threshold", type=float, default=DEFAULT_HISTORY_USE_THRESHOLD)
    p.add_argument("--low-confidence-fallback-threshold", type=float, default=LOW_CONFIDENCE_FALLBACK_THRESHOLD)
    p.add_argument("--disable-low-confidence-fallback", action="store_true")

    p.add_argument("--disable-bias", action="store_true")
    p.add_argument("--context-strategy", type=str, default="hybrid_long_context",
                   choices=["chunk_pack", "section_expand", "article_expand", "hybrid_long_context"])
    p.add_argument("--max-article-context-tokens", type=int, default=6000)
    p.add_argument("--max-section-context-tokens", type=int, default=2500)
    p.add_argument("--max-evidence-spans", type=int, default=12)
    p.add_argument("--disable-evidence-spans", action="store_true")
    p.add_argument("--answer-refined-query", action="store_true")

    args = p.parse_args()

    cfg = QAConfig(
        top_k=args.top_k,
        per_source_max=args.per_source_max,
        budget_tokens=args.budget_tokens,
        budget_words=args.budget_words,
        neighbor_headroom=args.neighbor_headroom,
        num_predict=args.num_predict,
        openai_model=args.openai_model,
        use_reranker=args.use_reranker,
        reranker_top_n=args.reranker_top_n,
        include_history=args.include_history,
        history_max_turns=args.history_max_turns,
        history_top_entries=args.history_top_entries,
        history_decay=args.history_decay,
        history_scale=args.history_scale,
        history_dist_penalty=args.history_dist_penalty,
        history_use_threshold=args.history_use_threshold,
        low_confidence_fallback_threshold=args.low_confidence_fallback_threshold,
        enable_low_confidence_fallback=not args.disable_low_confidence_fallback,
        retrieval_pool=args.retrieval_pool,
        per_source_pool=args.per_source_pool,
        final_limit=args.final_limit,
        use_bias=not args.disable_bias,
        context_strategy=args.context_strategy,
        max_article_context_tokens=args.max_article_context_tokens,
        max_section_context_tokens=args.max_section_context_tokens,
        max_evidence_spans=args.max_evidence_spans,
        include_evidence_spans=not args.disable_evidence_spans,
        answer_original_question=not args.answer_refined_query,
    )

    if args.q:
        res = agentic_run(args.q, cfg)
        print(res["answer"])
        print("\nSOURCES:")
        for s in res["citations"]:
            print(" -", s)
        print(
            f"\nRetrieval {res['retrieval_time']:.2f}s | "
            f"Generation {res['generation_time']:.2f}s | "
            f"Prompt tokens {res['prompt_tokens']} | "
            f"Output tokens {res['output_tokens']} | "
            f"Context tokens {res['context_tokens']} | "
            f"Question tokens {res['question_tokens']} | "
            f"Confidence {res['retrieval_confidence']:.2f}"
        )
        return

    print("MSK RAG ready. Type 'quit' to exit.")
    hist: List[Dict[str, str]] = []
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"quit", "exit"}:
            break
        if cfg.include_history:
            res = agentic_run(q, cfg=cfg, history=hist)
        else:
            res = agentic_run(q, cfg)
        print(res["answer"])
        print("\nSOURCES:")
        for s in res["citations"]:
            print(" -", s)
        print(
            f"\nRetrieval {res['retrieval_time']:.2f}s | "
            f"Generation {res['generation_time']:.2f}s | "
            f"Prompt tokens {res['prompt_tokens']} | "
            f"Output tokens {res['output_tokens']} | "
            f"Context tokens {res['context_tokens']} | "
            f"Question tokens {res['question_tokens']} | "
            f"Confidence {res['retrieval_confidence']:.2f}"
        )
        hist.extend([{"role": "user", "content": q}, {"role": "assistant", "content": res['answer']}])

if __name__ == "__main__":
    main()
