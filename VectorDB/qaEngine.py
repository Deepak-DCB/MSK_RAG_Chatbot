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
import json
import os
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import re

import dotenv
dotenv.load_dotenv()
from openai import OpenAI
import chromadb
import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# Embedding model constant (must match what was used to build chroma_store)
EMBED_MODEL = "text-embedding-3-large"



try:
    import tiktoken
except Exception:
    tiktoken = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(PROJECT_ROOT / "chroma_store")
COLLECTION_NAME = "msk_chunks"

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

    use_reranker: bool = True
    reranker_top_n: int = 10

    include_history: bool = False
    history_max_turns: int = 10
    history_top_entries: int = DEFAULT_HISTORY_TOP_ENTRIES
    history_decay: float = DEFAULT_HISTORY_DECAY
    history_scale: float = DEFAULT_HISTORY_SCALE
    history_dist_penalty: float = DEFAULT_HISTORY_DIST_PENALTY
    history_use_threshold: float = DEFAULT_HISTORY_USE_THRESHOLD

    use_bias: bool = True



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


# ── OpenAI client singleton ──────────────────────────────────────────────────

_openai_client: Optional[OpenAI] = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def openai_embed(texts: list[str]) -> list[list[float]]:
    """Embed one or more texts via the OpenAI embeddings API."""
    client = _get_openai_client()
    # Ensure all inputs are valid non-empty strings
    clean = [str(t).strip() if t else "empty" for t in texts]
    clean = [t if t else "empty" for t in clean]
    resp = client.embeddings.create(model=EMBED_MODEL, input=clean)
    return [d.embedding for d in resp.data]


class Backend:
    def __init__(self) -> None:
        self.collection = None

    def load_collection(self) -> Any:
        if self.collection is not None:
            return self.collection
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


def hybrid_search(
    question: str,
    collection,
    retrieval_pool: int = 50,
    rrf_k: int = 60,
) -> dict:
    """
    Hybrid search: dense (ChromaDB) + sparse (BM25) with Reciprocal Rank Fusion.
    Returns results in the same format as collection.query().
    """
    # Dense search
    q_emb = encode_query(question)
    dense_raw = collection.query(query_embeddings=q_emb, n_results=retrieval_pool)

    # BM25 search (if available)
    bm25_idx = BM25Index.get()
    bm25_idx._build(collection)
    bm25_results = bm25_idx.search(question, top_n=retrieval_pool)

    if not bm25_results:
        # BM25 unavailable — return dense-only results
        return dense_raw

    # RRF fusion
    doc_scores = {}  # text_hash -> {text, meta, rrf_score}

    # Score dense results
    dense_docs = dense_raw.get("documents", [[]])[0]
    dense_metas = dense_raw.get("metadatas", [[]])[0]
    dense_dists = dense_raw.get("distances", [[]])[0]

    for rank, (doc, meta, dist) in enumerate(zip(dense_docs, dense_metas, dense_dists)):
        key = hash(doc[:200])
        rrf = 1.0 / (rrf_k + rank + 1)
        if key in doc_scores:
            doc_scores[key]["rrf_score"] += rrf
        else:
            doc_scores[key] = {"text": doc, "meta": meta, "dist": float(dist), "rrf_score": rrf}

    # Score BM25 results
    for rank, item in enumerate(bm25_results):
        key = hash(item["text"][:200])
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
        answer, _, _ = ask_openai_llm(prompt, model=RERANKER_MODEL, num_predict=150)
        lines = [l.strip() for l in answer.strip().split("\n") if l.strip()]
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
    backend: Backend,
    openai_model: str,
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
        answer, _, _ = ask_openai_llm(prompt, model=RERANKER_MODEL, num_predict=256)
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


def build_prompt(question: str, context: List[Dict[str, Any]], history=None) -> str:
    ctx_block = format_context_block(context)

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

    return textwrap.dedent(f"""
        CONTEXT (internal, do not describe it explicitly to the user):
        ---
        {ctx_block}
        ---

        Now answer this question clearly and concisely as one integrated explanation.
        {instructions}

        Question: {question}
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
    answer, _, _ = ask_openai_llm(prompt, model=model, num_predict=8)
    letter = answer.strip().upper()
    return letter[0] if letter else "D"


def rewrite_query(user_q: str, category: str, openai_model: str, history=None) -> str:
    """
    Rewrite the query into an MSK-biomechanics-optimized form
    based on classification category A/B/C/D.
    Uses last 2 conversation turns to resolve pronouns and vague follow-ups.
    """
    # Build recent conversation context for the rewriter
    conv_context = ""
    if history:
        recent = history[-4:]  # last 2 turns (2 messages each: user + assistant)
        lines = []
        for turn in recent:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")[:200]  # truncate for rewriter
            lines.append(f"{role}: {content}")
        conv_context = "\n".join(lines)

    history_block = ""
    if conv_context:
        history_block = f"""\nRecent conversation (use to resolve pronouns like 'it', 'that', 'this'):
{conv_context}\n"""

    prompt = f"""
Rewrite the user's query into a more detailed MSK biomechanics retrieval query.
{history_block}
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
    refined, _, _ = ask_openai_llm(prompt, model=openai_model, num_predict=128)
    return refined.strip()


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
    "I'd like to help, but I need a bit more detail to give you an accurate, "
    "evidence-grounded answer.\n\n"
    "Could you tell me:\n\n"
    "1. **Where** exactly is the pain or discomfort? (e.g., left neck, right shoulder, lower back)\n"
    "2. **What does it feel like?** (e.g., sharp, dull ache, tingling, stiffness)\n"
    "3. **When** does it happen? (e.g., sitting at a desk, overhead reaching, after exercise)\n\n"
    "The more specific you are, the better I can match your concern to the "
    "MSK Neurology evidence base."
)

_FILLER_WORDS = {"i", "my", "me", "the", "a", "an", "is", "it", "have", "has",
                 "been", "am", "are", "was", "were", "do", "does", "did",
                 "hello", "hi", "hey", "help", "please", "thanks", "thank",
                 "some", "very", "really", "just", "also", "and", "or", "but",
                 "in", "on", "at", "to", "for", "of", "with", "from", "so",
                 "can", "could", "would", "should", "there", "here", "this",
                 "that", "what", "how", "why", "when", "where", "who"}


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
):
    cfg = cfg or QAConfig()

    # Step 0: check for vague queries — but NOT if there's conversation history
    # (follow-up questions like "when would I need surgery?" are contextual, not vague)
    if not history and _is_vague_query(question):
        clarification = _CLARIFICATION_RESPONSE
        if on_token:
            on_token(clarification)
        return {
            "answer": clarification,
            "citations": [],
            "contexts": [],
            "retrieval_confidence": 0.0,
            "retrieval_time": 0.0,
            "generation_time": 0.0,
            "prompt_tokens": 0,
            "output_tokens": 0,
            "context_tokens": 0,
            "question_tokens": count_tokens(question),
            "category": "clarification",
            "category_label": "Needs more detail",
            "refined_query": question,
        }

    # Step 1: classify
    category = classify_query(question, cfg.openai_model, history=history)

    # Step 2: rewrite for retrieval (with history for context-aware rewriting)
    refined_q = rewrite_query(question, category, cfg.openai_model, history=history)

    # Step 3: run run_qa() but forward history correctly
    if history:
        res = run_qa(refined_q, config=cfg, on_token=on_token,history=history)
    else:
        res = run_qa(refined_q, config=cfg, on_token=on_token)
    
    from qaEngine import CATEGORY_LABELS

    res["category"] = category
    res["category_label"] = CATEGORY_LABELS.get(category, "Unknown")
    res["refined_query"] = refined_q
    return res



def _truncate_history(history, max_turns=5, max_chars_per_msg=800, max_total_tokens=2500):
    """
    Prepare conversation history for the LLM messages array.
    - Keeps the last `max_turns` pairs (10 messages max)
    - Truncates each message to `max_chars_per_msg` characters
    - Stops adding once estimated token budget is reached
    """
    if not history:
        return []

    # Take the last N messages (max_turns * 2 for user+assistant pairs)
    recent = history[-(max_turns * 2):]

    truncated = []
    total_tokens = 0

    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Truncate long messages
        if len(content) > max_chars_per_msg:
            content = content[:max_chars_per_msg] + "…"

        est_tokens = len(content) // 4  # rough estimate: 1 token ≈ 4 chars
        if total_tokens + est_tokens > max_total_tokens:
            break

        total_tokens += est_tokens
        truncated.append({"role": role, "content": content})

    return truncated


def ask_openai_llm(prompt: str, model: str, num_predict: int, on_token=None, history=None):
    """
    Clean, stable Chat Completions wrapper for GPT-4.1 models.
    - No Responses API
    - Guaranteed correct formatting/newlines
    - Streams tokens cleanly via delta.content
    - Fallback to non-streaming if streaming fails
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")

    client = OpenAI(api_key=api_key)

    # Token counting for telemetry
    prompt_tokens = count_tokens(prompt)

    # Build multi-turn messages array
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject conversation history for multi-turn context
    conv_history = _truncate_history(history)
    if conv_history:
        messages.extend(conv_history)

    messages.append({"role": "user", "content": prompt})

    parts = []

    # GPT-5 family: pin reasoning effort so the output-token budget goes to the answer
    extra_args = {"reasoning_effort": "none"} if model.startswith("gpt-5") else {}

    # ---------- 1) Try streaming first ----------
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_completion_tokens=num_predict,
            **extra_args,
        )

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
        # Only fall back for streaming-specific issues, not API errors
        err_str = str(stream_err)
        if "invalid" in err_str.lower() or "400" in err_str:
            raise RuntimeError(f"OpenAI API error: {stream_err}")
        # Fall back to non-streaming for other issues


    # ---------- 2) Non-streaming fallback ----------
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            max_completion_tokens=num_predict,
            **extra_args,
        )
        content = resp.choices[0].message.content or ""
        answer = content.strip()
        output_tokens = count_tokens(answer)
        return answer, int(prompt_tokens), int(output_tokens)

    except Exception as e:
        raise RuntimeError(f"Chat completions failed: {e}")










def run_qa(
    question: str,
    config: Optional[QAConfig] = None,
    *,
    on_token: Optional[Callable[[str], None]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    cfg = config or QAConfig()

    effective_budget_tokens = (
        cfg.budget_tokens if cfg.budget_tokens and cfg.budget_tokens > 0
        else words_to_tokens_heuristic(cfg.budget_words)
    )

    coll = _backend.load_collection()

    t0 = time.time()

    # ---- Hybrid search (dense + BM25) with multi-query ----
    raw = hybrid_search(question, coll, retrieval_pool=cfg.retrieval_pool)

    # Multi-query: generate 2 reformulations for broader coverage
    alt_queries = generate_multi_queries(question, n=2)
    for alt_q in alt_queries:
        alt_raw = hybrid_search(alt_q, coll, retrieval_pool=cfg.retrieval_pool // 2)
        # Merge into main results (dedup by text hash)
        seen = set(hash(d[:200]) for d in raw.get("documents", [[]])[0])
        for doc, meta, dist in zip(
            alt_raw.get("documents", [[]])[0],
            alt_raw.get("metadatas", [[]])[0],
            alt_raw.get("distances", [[]])[0],
        ):
            key = hash(doc[:200])
            if key not in seen:
                seen.add(key)
                raw["documents"][0].append(doc)
                raw["metadatas"][0].append(meta)
                raw["distances"][0].append(dist)

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
        }

    # ---- compute retrieval confidence (TIER1) ----
    try:
        dists = [float(d) for d in (raw.get("distances", [[1.0]])[0] or [])]
    except Exception:
        dists = [1.0]
    if not dists:
        dists = [1.0]

    k = min(5, len(dists))
    corpus_confidence = float(np.mean([1.0 - d for d in dists[:k]]))
    retrieval_confidence = corpus_confidence  # TIER1

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
        print(f"[RERANKER] model={cfg.openai_model} | top_n={cfg.reranker_top_n} | use={cfg.use_reranker}")
        for src, group in list(grouped.items()):
            grouped[src] = maybe_rerank(
            question,
            group,
            _backend,
            cfg.openai_model,
            cfg.reranker_top_n
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
        }

    # prompt + generation
    prompt = build_prompt(question, context, history=history)
    context_tokens = sum(_chunk_tokens(it) for it in context)
    question_tokens = count_tokens(question)

    if cfg.generate_answer:
        t1 = time.time()
        first_token_time = None

        def token_callback(tok):
            nonlocal first_token_time
            if first_token_time is None:
                first_token_time = time.time()
            if on_token:
                on_token(tok)

        answer_text, prompt_tokens, output_tokens = ask_openai_llm(
            prompt,
            model=cfg.openai_model,
            num_predict=cfg.num_predict,
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
        "retrieval_time": retrieval_time,
        "generation_time": gen_time,
        "first_token_latency": first_token_latency,
        "prompt_tokens": prompt_tokens,      # computed manually
        "output_tokens": output_tokens,      # computed manually
        "context_tokens": context_tokens,
        "question_tokens": question_tokens,
        "citations": uniq,
        "retrieval_confidence": float(retrieval_confidence),
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

    p.add_argument("--disable-bias", action="store_true")

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
        retrieval_pool=args.retrieval_pool,
        per_source_pool=args.per_source_pool,
        final_limit=args.final_limit,
        use_bias=not args.disable_bias,
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
