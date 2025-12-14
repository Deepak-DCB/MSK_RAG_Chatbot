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
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer



try:
    import tiktoken
except Exception:
    tiktoken = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIR = str(PROJECT_ROOT / "chroma_store")
COLLECTION_NAME = "msk_chunks"

OPENAI_MODEL = "gpt-4.1-mini"

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
- The seven-section structure is recommended when the retrieved context supports a structured clinical explanation; it is not an absolute requirement.
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


def detect_embed_model() -> str:
    alt_txt = PROJECT_ROOT / "embeddings" / "embedding_model.txt"
    if alt_txt.exists():
        return alt_txt.read_text(encoding="utf-8").strip()
    return "mixedbread-ai/mxbai-embed-large-v1"


class Backend:
    def __init__(self) -> None:
        self.embedder: Optional[SentenceTransformer] = None
        self.collection = None


    def load_embedder(self, model_name: Optional[str] = None) -> SentenceTransformer:
        if self.embedder is not None:
            return self.embedder
        model_name = model_name or detect_embed_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(model_name, device=device)
        _ = self.embedder.encode(["warmup"])
        return self.embedder

    def load_collection(self) -> Any:
        if self.collection is not None:
            return self.collection
        import os
        os.environ["CHROMA_DATA_PATH"] = PERSIST_DIR
        try:
            client = chromadb.PersistentClient(path=PERSIST_DIR)
            self.collection = client.get_or_create_collection(COLLECTION_NAME)
        except Exception as e:
            chunks_path = PROJECT_ROOT / "MSKArticlesINDEX" / "chunks.parquet"
            emb_path = PROJECT_ROOT / "embeddings" / "embeddings.npy"
            if not emb_path.exists():
                raise RuntimeError(f"embeddings.npy not found at {emb_path}") from e
            chunks = pd.read_parquet(chunks_path)
            embs = np.load(emb_path)
            client = chromadb.Client()
            self.collection = client.create_collection(COLLECTION_NAME)
            self.collection.add(
                embeddings=embs.tolist(),
                documents=chunks["embed_text"].tolist(),
                metadatas=chunks.drop(columns=["embed_text"]).to_dict(orient="records"),
                ids=[str(i) for i in range(len(chunks))],
            )
        return self.collection




_backend = Backend()


def encode_query(text: str, model: SentenceTransformer) -> np.ndarray:
    return model.encode(text, normalize_embeddings=True)

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
    Batch LLM-based reranker: sends one prompt containing all candidates and returns ranked top_n.
    This replaces per-candidate calls to llm_rerank_score.
    """
    if not candidates:
        return candidates

    # Build enumerated list of short candidate excerpts (truncate to safe length to keep prompt small)
    items = []
    for i, c in enumerate(candidates, start=1):
        text = c["text"]
        # keep excerpt short for scoring prompt; preserve enough context
        excerpt = text.replace("\n", " ").strip()
        items.append((i, excerpt, c))

    prompt_lines = [
        "You are scoring how relevant each of the following retrieved text chunks is for answering the query.",
        "Return a comma-separated list of numbers (one score per item) from 0 to 10 matching item order. No commentary.",
        "",
        f"User query:",
        question,
        "",
        "Chunks:"
    ]
    for i, excerpt, _ in items:
        prompt_lines.append(f"{i}. {excerpt}")

    prompt = "\n".join(prompt_lines)

    answer, _, _ = ask_openai_llm(prompt, model=openai_model, num_predict=512)
    # Expect answers like: "8, 6.5, 0, 7, 4" or lines "1: 8\n2: 6.5\n..."
    text = (answer or "").strip()
    scores = []
    # try parsing robustly
    # first try comma-separated floats
    try:
        if "," in text:
            tokens = [t.strip() for t in text.split(",")]
            for i, (_, _, c) in enumerate(items):
                val = float(tokens[i]) if i < len(tokens) else 0.0
                scores.append((c, val))
        else:
            # fallback: parse any floats in order
            floats = [float(m) for m in re.findall(r"[-+]?\d*\.\d+|\d+", text)]
            for i, (_, _, c) in enumerate(items):
                val = floats[i] if i < len(floats) else 0.0
                scores.append((c, val))
    except Exception:
        # if parsing fails, fallback to original distance
        return sorted(candidates, key=lambda x: x["dist"])[:max(1, min(top_n,len(candidates)))]

    # convert score -> distance (lower = better)
    reranked = []
    for c, val in scores:
        val = max(0.0, min(10.0, val))
        dist = 1.0 - (val / 10.0)
        reranked.append({"text": c["text"], "meta": c["meta"], "dist": dist})

    reranked.sort(key=lambda x: x["dist"])
    return reranked[:max(1, min(top_n, len(reranked)))]




def select_relevant_history(
    history: List[Dict[str, str]],
    query: str,
    embedder: SentenceTransformer,
    *,
    max_turns: int,
    max_entries: int,
    decay_factor: float,
    scale: float,
    dist_penalty: float,
) -> List[Dict[str, Any]]:
    if not history or embedder is None:
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

    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    h_texts = [f"[{role.upper()}] {text}" for _, role, text in entries]
    h_embs = embedder.encode(h_texts, normalize_embeddings=True)

    sims = (h_embs @ q_emb).tolist()
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


def build_prompt(question: str, context: List[Dict[str, Any]]) -> str:
    ctx_block = format_context_block(context)

    # Simple heuristic: if the user asks about exercises/timing/frequency, use concise format
    simple_q_re = re.compile(
        r'\b(exercise|exercises|when to|how often|how long|frequency|reps?|sets?|timing|dose|doseing)\b',
        re.I,
    )
    use_sections = not bool(simple_q_re.search(question))

    if use_sections:
        instructions = """
        INSTRUCTIONS:
        - Return the answer formatted in Markdown (headers, short paragraphs, numbered lists where appropriate).
        - You MAY use the recommended 7-section structure; a brief 1–2 sentence intro before the sections is allowed.
        - Do NOT mention or quote the internal context or say "based on the context".
        - If the context is insufficient, state: "Insufficient evidence in the supplied context."
        - Keep the answer concise.
        """
    else:
        instructions = """
        INSTRUCTIONS:
        - Return a short, practical Markdown answer (no 7-section structure).
        - Provide 3–6 numbered, actionable steps and a 1–2 sentence rationale.
        - If the context is insufficient, say "Insufficient evidence in the supplied context."
        - Keep it concise and focused on timing/dosing/pacing.
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




def classify_query(user_q: str, model: str ) -> str:
    """
    Agentic pre-step: classify the type of question so the RAG
    knows what retrieval domain to target.
    """
    prompt = f"""
        You will classify the user's query so that a biomechanical RAG system can retrieve the correct type of sections.

        User query:
        "{user_q}"

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


def rewrite_query(user_q: str, category: str, openai_model: str) -> str:
    """
    Rewrite the query into an MSK-biomechanics-optimized form
    based on classification category A/B/C/D.
    """
    prompt = f"""
Rewrite the user's query into a more detailed MSK biomechanics retrieval query.

Original:
"{user_q}"

Category = {category}

Rules:
- If A: emphasize benign muscular/postural mechanisms, fatigue, suboccipitals, levator, trapezius, strain patterns.
- If B: emphasize specific MSKNeurology biomechanical drivers (scapular orientation, plexus traction, rib mechanics, etc.)
- If C: emphasize neurovascular or red-flag patterns.
- If D: rewrite neutrally with maximal biomechanical detail.

Return ONLY the rewritten query, no commentary.
"""
    refined, _, _ = ask_openai_llm(prompt, model=openai_model, num_predict=128)
    return refined.strip()


def agentic_run(
    question: str,
    cfg: Optional[QAConfig] = None,
    history=None,
    on_token = None,
):
    cfg = cfg or QAConfig()
    # Step 1: classify
    category = classify_query(question, cfg.openai_model)

    # Step 2: rewrite for retrieval
    refined_q = rewrite_query(question, category, cfg.openai_model)

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



def ask_openai_llm(prompt: str, model: str, num_predict: int, on_token=None):
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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",  "content": prompt}
    ]

    parts = []

    # ---------- 1) Try streaming first ----------
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_completion_tokens=num_predict,
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

    except Exception:
        pass  # fall back to non-streaming


    # ---------- 2) Non-streaming fallback ----------
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            max_completion_tokens=num_predict,
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

    embedder = _backend.load_embedder()
    coll = _backend.load_collection()

    t0 = time.time()
    q_emb = encode_query(question, embedder)
    raw = coll.query(query_embeddings=q_emb, n_results=cfg.retrieval_pool)
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
                embedder,
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
    prompt = build_prompt(question, context)
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
