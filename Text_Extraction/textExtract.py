#!/usr/bin/env python3
"""
textExtract.py (v8.6.1 — sentence-first with hard token ceiling, thread-safe)
Token-aware, causal-grouped, parallel chunker for MSK Neurology HTTrack mirrors.

Improvements vs 8.5.1:
- Optional hard token ceiling in sentence mode (eliminates long-tail outliers)
- Adaptive overlap suppression for near-ceiling chunks
- Long-sentence splitting by clauses (fallback to words)
- Thread-safe config: no runtime mutation of module-level globals
- Deterministic article_seq ordering preserved

Outputs:
- all_articles.jsonl  (article metadata)
- chunks.parquet      (chunk rows with `embed_text`, metadata, images JSON)
  *Falls back to chunks.jsonl if Parquet engine unavailable.*
"""

from __future__ import annotations

# --- Chunking Strategy ---
DEFAULT_EMBED_TOKEN_LIMIT = 0
DEFAULT_OVERLAP_RATIO     = 0.25
DEFAULT_MAX_SENTENCES     = 6
DEFAULT_SENT_OVERLAP      = 2
DEFAULT_MIN_WORDS         = 50

# --- Quality Thresholds ---
MIN_CHUNK_WORD_THRESHOLD  = 50
CAUSAL_GROUP_MAX_WORDS    = 350

# --- Token Estimation ---
TOKEN_TO_WORD_APPROX      = 0.75
DEFAULT_MODEL_TOKENIZER   = "o200k_base"

# --- Token ceilings (sentence mode) ---
DEFAULT_MAX_TOKENS_PER_CHUNK     = 512
DEFAULT_ADAPTIVE_OVERLAP_THRESH  = 350
DEFAULT_LONG_SENT_TOKEN_THRESH   = 200

# --- Image Filtering ---
MIN_IMG_DIM     = 48
MIN_IMG_AREA    = 3000
PLACEHOLDER_PREFIX = "data:image/svg"

# --- HTML Element Filtering ---
TAG_BLOCK = {"script", "style", "noscript", "iframe", "footer", "header", "nav", "aside"}
IGNORES   = ("/feed/", "/wp-admin/", "/wp-includes/", "/wp-content/")

# --- CSS Selectors for main content ---
CONTENT_SELECTORS = [
    ("div", {"class": "entry-content"}),
    ("article", {}),
    ("main", {}),
    ("div", {"id": "content"}),
]

META_PROPS = {
    "published": "article:published_time",
    "updated":   "article:modified_time",
    "url":       "og:url",
    "title_og":  "og:title",
}

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup, Tag

# Optional dependencies
try:
    import chardet
except Exception:
    chardet = None

try:
    import ftfy
except Exception:
    ftfy = None

try:
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
        _NLTK_HAS_PUNKT = True
    except LookupError:
        _NLTK_HAS_PUNKT = False
    if _NLTK_HAS_PUNKT:
        sent_tokenize = nltk.sent_tokenize
    else:
        raise Exception("NLTK punkt missing")
except Exception:
    def sent_tokenize(text: str) -> List[str]:
        return re.split(r'(?<=[.!?])\s+(?=[A-Z(])', text)

try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ------------------------- STRICT BLOCK-LEVEL REFERENCE DETECTOR -------------------------

# Matches reference-list style blocks
IS_REFERENCE_BLOCK_RE = re.compile(
    r"""
    # Pattern 1 — Journal-style year + volume/issue/pages
    (?:\b(?:19|20)\d{2}\b                  # a year like 2018
        .*?(?:;|\(|:)                      # followed by ;  or (  or :
        .{0,40}?                           # some journal formatting text
        \b\d{1,4}\b                        # a page/issue/volume number 1–4 digits
    )

    |

    # Pattern 2 — "et al." anywhere
    (?:\bet\ al\.\b)

    |

    # Pattern 3 — DOI
    (?:\bdoi\s*[:=]\s*\S+)

    |

    # Pattern 4 — PMID
    (?:\bpmid\s*[:=]\s*\d+)

    |

    # Pattern 5 — Typical reference with two+ surnames and a year
    (?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+     # "Musa A Farhan SA"   or "Smith John"
        ,\s*\b(?:19|20)\d{2}\b)            # ", 2019"

    """,
    re.IGNORECASE | re.VERBOSE
)

# -------------------------------------------------------------------------------------------

_JOUR_HINT = r"(?:Curr|J|Jour|Journal|Clin|Arch|Neurol|Spine|Orthop|Otol|Rheum|Med|Ther)"

CAUSAL_CUES_RE = re.compile(
    r'\b(because|since|therefore|hence|thus|so that|leads to|results in|due to|as a result)\b',
    re.I
)

CIT_INLINE_RE = re.compile("|".join([
    r"\bdoi\s*[:=]\s*\S+",
    r"\bpmid\s*[:=]\s*\d+",
    r"\bpmcid\s*[:=]\s*\S+",
    r"\(\s*[A-Z][a-z]+(?:\s+et\s+al\.)?,?\s*(?:,?\s*(?:19|20)\d{2})\s*\)",
    r"–\s*[A-Z][a-z]+(?:\s+et\s+al\.)?,?\s*(?:,?\s*(?:19|20)\d{2})",
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*(?:et\s+al\.)?,?\s*(?:,?\s*(?:19|20)\d{2})",
    r"\b(J|Jour|Journal|Spine|Med|Clin|Rev|Arch|Neurol)\.?[\sA-Z]*\d{4};?\s*\d*\(?.*?\)?[:;]?\d*[-–]?\d*",
    rf"\bReferences?:\s+[A-Z].{{0,160}}?\b{_JOUR_HINT}\b[^\.]*?\d{{1,4}}[-–]\d{{1,4}}\.?"
]), re.I)

CAPTION_RE   = re.compile(r"\bfig(?:ure)?\b|\bsource:|click|©", re.I)
REFERENCE_RE = re.compile(r"^(references|bibliography)\b", re.I)

# NEW: parenthetical citation scrubber (for ugly "(; ; ; , ...)" blobs)
PAREN_CIT_RE = re.compile(
    r"""
    \(
      [^)]{0,200}?                                     # reasonable length
      (?:et\ al\.|doi|pmid|\b(?:19|20)\d{2}\b|[A-Z][a-z]+)  # citation-ish hints
      [^)]*?
    \)
    """,
    re.VERBOSE | re.IGNORECASE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("textExtract")

# ----------------------------- Config & tokenizer --------------------------------
@dataclass(frozen=True)
class Config:
    embed_token_limit: int = DEFAULT_EMBED_TOKEN_LIMIT
    overlap_ratio: float   = DEFAULT_OVERLAP_RATIO
    max_sentences: int     = DEFAULT_MAX_SENTENCES
    sent_overlap: int      = DEFAULT_SENT_OVERLAP
    min_words: int         = DEFAULT_MIN_WORDS

    min_chunk_word_threshold: int = MIN_CHUNK_WORD_THRESHOLD
    causal_group_max_words: int   = CAUSAL_GROUP_MAX_WORDS

    model_tokenizer: str = DEFAULT_MODEL_TOKENIZER

    max_tokens_per_chunk: int      = DEFAULT_MAX_TOKENS_PER_CHUNK
    adaptive_overlap_threshold: int = DEFAULT_ADAPTIVE_OVERLAP_THRESH
    long_sent_token_threshold: int  = DEFAULT_LONG_SENT_TOKEN_THRESH


def fix_mojibake(text: str) -> str:
    """Fix common UTF-8/Win-1252 artifacts and double-encoded diacritics.
    Prefers ftfy when available; falls back to regex + replacement map.
    """
    if not text:
        return text

    # Try ftfy if installed; this already fixes many encoding issues
    if ftfy is not None:
        try:
            return ftfy.fix_text(text)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Step 1: Common single-pass substitutions (Win-1252 → UTF-8)
    # ------------------------------------------------------------------
    replacements = {
        "â€“": "–", "â€”": "—",
        "â€˜": "'", "â€™": "'",
        "â€œ": '"', "â€": '"',
        "â€¦": "…",
        "Â ": "", "Â": "",
        "â€¢": "•",
        "â€ ": "– ",
        "‚Ä´": "'", "‚Ä¬": "–",
        "‚Ä®": "–", "‚Äº": '"', "‚Ä¹": '"',
        "�": "",
        "√©": "é", "√®": "è", "√¢": "â", "√ª": "ê", "√±": "ñ",
        "‚Ä´s": "'s",
        # extra cleanup of rare artifacts seen in chunks
        "¬": "",         # stray not-signs around quotes/words
        "́": "",         # combining acute accent left behind
        "’Â": "’",       # double-encoded apostrophe patterns
        "Â’": "’",
        "Â«": "«",
        "Â»": "»",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # ------------------------------------------------------------------
    # Step 2: Deep sweep for residual mojibake byte sequences
    # Removes leftover multi-byte corruption fragments like Ã, â, Â, √, etc.
    # ------------------------------------------------------------------
    text = re.sub(r"(?:Ã.|â.|Â.|√.|�|‚Ä.|â€¦|â€¢)+", "", text)

    # ------------------------------------------------------------------
    # Step 3: Targeted repair of observed double-encoded diacritics
    # ------------------------------------------------------------------
    text = (
        text.replace("√¶", "ö")
            .replace("√ö", "ö")
            .replace("√õ", "ö")
            .replace("√ñ", "Ñ")
            .replace("√ò", "ò")
            .replace("√ó", "ó")
            .replace("√ú", "ú")
            .replace("√ü", "ü")
            .replace("√à", "à")
            .replace("√á", "á")
            .replace("√", "")       # remove stray '√' if not part of pair
            .replace("Ä쬆", "Á")    # fixes “Ä쬆Rold√°n” → “ÁRoldán”
            .replace("Ä", "A")      # generic fallback
    )

    # ------------------------------------------------------------------
    # Step 4: Normalize and tidy whitespace
    # ------------------------------------------------------------------
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# ----------------------------- Token counter ------------------------------------
class TokenCounter:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.enc = None
        if tiktoken is not None:
            try:
                self.enc = tiktoken.get_encoding(cfg.model_tokenizer)
            except Exception:
                try:
                    self.enc = tiktoken.encoding_for_model(cfg.model_tokenizer)
                except Exception:
                    self.enc = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.enc is None:
            return int(len(text.split()) / TOKEN_TO_WORD_APPROX)
        try:
            return len(self.enc.encode(text))
        except Exception:
            return int(len(text.split()) / TOKEN_TO_WORD_APPROX)

# ----------------------------- IO helpers ---------------------------------------
def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sniff_html(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            head = fh.read(1024).lower()
        return b"<html" in head or b"<!doctype html" in head
    except Exception:
        return False

def list_index_files(site_root: Path) -> List[Path]:
    hits = []
    for filepath in glob.glob(str(site_root / "**" / "index*"), recursive=True):
        normalized = filepath.lower().replace("\\", "/")
        if any(ig in normalized for ig in IGNORES):
            continue
        candidate = Path(filepath)
        if candidate.is_file() and sniff_html(candidate):
            hits.append(candidate)
    return hits

def detect_site_root(root: Path) -> Path:
    root = root.resolve()
    if list_index_files(root):
        return root
    candidate = root / "mskneurology.com"
    if candidate.exists() and list_index_files(candidate):
        return candidate
    for sub in root.iterdir():
        if sub.is_dir() and list_index_files(sub):
            return sub
    return root

def read_html(path: Path) -> BeautifulSoup:
    raw = path.read_bytes()
    text = None
    if chardet is not None:
        try:
            enc = chardet.detect(raw).get("encoding") or "utf-8"
            text = raw.decode(enc, errors="replace")
        except Exception:
            pass
    if text is None:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")
    text = unicodedata.normalize("NFKC", text)
    text = fix_mojibake(text)
    try:
        return BeautifulSoup(text, "lxml")
    except Exception:
        return BeautifulSoup(text, "html.parser")

def clean_text(fragment: Tag) -> str:
    soup = BeautifulSoup(str(fragment), "lxml")
    for bad in soup.find_all(lambda t: t.name in TAG_BLOCK):
        bad.decompose()
    return fix_mojibake(soup.get_text(" ", strip=True))

# ----------------------------- Image helpers ------------------------------------
def _real_img_src(img: Tag) -> Optional[str]:
    for attr in ("src", "data-src", "data-lazy-src", "data-original"):
        val = img.get(attr)
        if val and not str(val).startswith(PLACEHOLDER_PREFIX):
            return str(val)
    for attr in ("srcset", "data-srcset"):
        val = img.get(attr)
        if val:
            first = val.split(",")[0].split()[0]
            if first and not first.startswith(PLACEHOLDER_PREFIX):
                return first
    return None

def _dims_from_url(src: str):
    m = re.search(r"-([1-9]\d*)x([1-9]\d*)\.(?:jpg|jpeg|png|gif|webp)", src, re.I)
    return (int(m[1]), int(m[2])) if m else (None, None)

def _is_tiny_image(img: Tag, src: str) -> bool:
    def to_int(x):
        try:
            return int(x or 0)
        except Exception:
            return 0
    w = to_int(img.get("width"))
    h = to_int(img.get("height"))
    if not w or not h:
        wu, hu = _dims_from_url(src)
        w, h = w or wu or 0, h or hu or 0
    return bool(w and h and (min(w, h) < MIN_IMG_DIM or (w * h) < MIN_IMG_AREA))

def _caption_text(img: Tag) -> str:
    if img.parent and img.parent.name == "figure":
        cap = img.parent.find("figcaption")
        if cap:
            return fix_mojibake(cap.get_text(" ", strip=True))
    return ""

def _clean_caption(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^\s*(source|credit)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s*[-–—:;]*\s*$", "", text)
    return fix_mojibake(text)

def _merge_img_strings(img: Tag) -> Dict[str, str]:
    cap   = _clean_caption(_caption_text(img))
    alt   = fix_mojibake((img.get("alt") or "").strip())
    title = fix_mojibake((img.get("title") or "").strip())
    merged = " ".join(dict.fromkeys([x for x in (cap, alt, title) if x]))
    return {"caption": cap, "alt": alt, "title": title, "image_text": merged}

def find_images(node: Tag) -> List[Dict[str, str]]:
    out = []
    for img in node.find_all("img"):
        src = _real_img_src(img)
        if not src:
            continue
        if _is_tiny_image(img, src):
            continue
        out.append({"src": src, **_merge_img_strings(img)})
    return out

# ----------------------------- Text cleanup & citations --------------------------
def strip_inline_citations(text: str) -> str:
    if not text:
        return text

    # --- Remove MSK-style superscript leftovers (safe patterns) ---
    # Pattern 1: comma-separated numeric citation clusters: "2 , 7 , 19 , 21"
    text = re.sub(r"\b(?:\d+\s*,\s*)+\d+\b", "", text)

    # Pattern 2: weakly spaced numeric citation clusters: "2 7 19 21"
    # (kept conservative to avoid obvious years/measurements)
    text = re.sub(
        r"\b(?:(?<!\d\d)(?<!\d\.)\d{1,2}\s+){1,6}\d{1,2}\b",
        "",
        text
    )

    # Remove MSK-style leading author attributions: "– Smith &", "– Kö", "– Swift &", etc.
    text = re.sub(
        r"^\s*[–\-]\s*[A-ZÅÄÖØÆËÉáàâäãåçéèêëíìîïñóòôöõúùûüÿćčžšß][A-Za-zÅÄÖØÆËÉáàâäãåçéèêëíìîïñóòôöõúùûüÿćčžšß]*(?:\s*&)?\s*",
        "",
        text
    )

    # NEW: drop obviously citation-only parentheses with mostly punctuation/digits
    def _drop_paren(m: re.Match) -> str:
        inner = m.group(0)
        payload = re.sub(r"\s", "", inner[1:-1])
        if not payload:
            return ""
        # remove punctuation/digits; if little remains, it's likely citation noise
        noise = re.sub(r"[;:,&0-9\-\.\']", "", payload)
        return "" if len(noise) <= 3 else inner

    text = PAREN_CIT_RE.sub(_drop_paren, text)

    # Existing inline citation removal
    text = CIT_INLINE_RE.sub("", text)

    # Normalize whitespace
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def prune_trailing_biblio(text: str, min_words: int) -> str:
    if not text or len(text) < 60:
        return text

    patterns = [
        r"(?:[A-Z][a-z]+(?:\s+[A-Z]\.){0,2},?\s*){2,}(?:19|20)\d{2}",
        r"(?:19|20)\d{2};\d+\(\d+\):\d+(?:-\d+)?",
        r"\b(et al\.|Elsevier|Springer|Wiley|Oxford|Lippincott)\b",
        r"–\s*[A-Z][a-z]+,\s*(?:19|20)\d{2}",
        # Explicit “References …” tail
        r"\bReferences\b.*$",
        r"\s*\([^)]*?(?:19|20)\d{2}[^)]*?\)\s*$",
        r"\bReferences?:[^\.]{0,200}\.",
        rf"\b[A-Z][a-z]{{2,}}\s(?:Opin|Orthop|Rheum|Neurol|Otol|Med|Ther)\b[^\.]*?\d{{1,4}}[-–]\d{{1,4}}\.?$",
    ]
    pat = re.compile("|".join(patterns))
    m = pat.search(text)
    if m:
        text = text[:m.start()].rstrip(" ,;.:–-")
    return text if len(text.split()) >= int(min_words * 0.5) else ""

# ----------------------------- Grouping & windowing -----------------------------
def causal_group(sentences, max_group_words=CAUSAL_GROUP_MAX_WORDS):
    groups = []
    buf = []
    wc = 0
    for s in sentences:
        s_wc = len(s.split())
        if not buf:
            buf = [s]
            wc = s_wc
            continue
        if (CAUSAL_CUES_RE.search(s) or CAUSAL_CUES_RE.search(buf[-1])) and (wc + s_wc) <= max_group_words:
            buf.append(s)
            wc += s_wc
        else:
            groups.append(" ".join(buf))
            buf = [s]
            wc = s_wc
    if buf:
        groups.append(" ".join(buf))
    return groups

def window_by_tokens(text, tok, max_tokens, overlap_ratio):
    if not text.strip():
        return []
    if tiktoken and tok.enc:
        tokens = tok.enc.encode(text)
        win = max_tokens
        step = max(1, int(win * (1 - overlap_ratio)))
        out = []
        i = 0
        while i < len(tokens):
            j = min(len(tokens), i + win)
            chunk = tok.enc.decode(tokens[i:j])
            if chunk.strip():
                out.append((chunk, i, j))
            i += step
        return out

    # fallback
    words = text.split()
    win = max_tokens
    step = max(1, int(win * (1 - overlap_ratio)))
    out = []
    i = 0
    while i < len(words):
        j = min(i + win, len(words))
        chunk = " ".join(words[i:j]).strip()
        if chunk:
            out.append((chunk, i, j))
        i += step
    return out

def window_by_sentences(sentences, max_sentences, overlap):
    n = len(sentences)
    if n == 0:
        return []
    step = max(1, max_sentences - overlap)
    out = []
    i = 0
    while i < n:
        j = min(i + max_sentences, n)
        out.append(" ".join(sentences[i:j]))
        i += step
    return out

# ----------------------------- Hard-cap helpers -----------------------------
_CLAUSE_SPLIT_RE = re.compile(r'(?<=[:;,\—\-\)])\s+|(?<=\.)\s+(?=[A-Z0-9])')

def _clause_split(sentence):
    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(sentence) if p.strip()]
    if not parts:
        parts = [s.strip() for s in sentence.split(',') if s.strip()]
    return parts if parts else [sentence]

def _pack_sentences_to_hard_cap(sentences, tok, cfg):
    if cfg.max_tokens_per_chunk <= 0:
        return [" ".join(sentences)]

    tcache = {}

    def tcount(x):
        if x in tcache:
            return tcache[x]
        v = tok.count(x)
        tcache[x] = v
        return v

    out = []
    cur = []
    cur_t = 0
    cap = cfg.max_tokens_per_chunk

    for s in sentences:
        st = tcount(s)
        if st > cap:
            for c in _clause_split(s):
                ct = tcount(c)
                if ct > cap:
                    words = c.split()
                    wcur = []
                    for w in words:
                        cand = " ".join(wcur + [w])
                        if tcount(cand) > cap:
                            if wcur:
                                out.append(" ".join(wcur))
                            wcur = [w]
                        else:
                            wcur.append(w)
                    if wcur:
                        rem = " ".join(wcur)
                        rct = tcount(rem)
                        if cur_t + rct <= cap:
                            cur.append(rem)
                            cur_t += rct
                        else:
                            out.append(rem)
                            cur = []
                            cur_t = 0
                else:
                    if cur_t + ct <= cap:
                        cur.append(c)
                        cur_t += ct
                    else:
                        if cur:
                            out.append(" ".join(cur))
                        cur = [c]
                        cur_t = ct
            continue

        if cur_t + st <= cap:
            cur.append(s)
            cur_t += st
        else:
            if cur:
                out.append(" ".join(cur))
            cur = [s]
            cur_t = st

    if cur:
        out.append(" ".join(cur))
    return out

def _apply_adaptive_overlap(chunks, tok, cfg, base_overlap_sent=1):
    if base_overlap_sent <= 0:
        return chunks
    out = []
    for i, text in enumerate(chunks):
        sents = sent_tokenize(text)
        tok_len = tok.count(text)
        use = 0 if tok_len >= cfg.adaptive_overlap_threshold else base_overlap_sent
        if i > 0 and use > 0:
            prev = sent_tokenize(chunks[i - 1])
            prefix = prev[-use:] if len(prev) >= use else prev
            out.append(" ".join(prefix + sents))
        else:
            out.append(" ".join(sents))
    return out

# ----------------------------- Flush context -------------------------------------
@dataclass
class FlushContext:
    paragraph_buffer: List[str] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    output_chunks: List[Dict[str, Any]] = field(default_factory=list)
    current_section: str = "Main"
    article_seq: int = 0

# ----------------------------- Metadata extraction -------------------------------
def extract_meta(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    title_tag = soup.find("h1")
    title = (
        title_tag.get_text(strip=True)
        if title_tag else (soup.title.get_text(strip=True) if soup.title else None)
    )
    meta = {
        "title": title,
        "title_og": None,
        "url": None,
        "published": None,
        "updated": None,
    }
    for key, prop in META_PROPS.items():
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            meta[key] = tag["content"].strip()
    return meta

def find_main_content(soup: BeautifulSoup) -> Tag:
    for name, attrs in CONTENT_SELECTORS:
        el = soup.find(name, attrs=attrs)
        if el:
            return el
    return soup.body or soup

# ----------------------------- Block materialization -----------------------------
def _materialize_chunks_for_block(
    *,
    block_text,
    header_title,
    section,
    cfg,
    tok,
    img_texts,
    images_for_block,
    article_id,
    article_seq_start,
    source_relpath
):
    sentences = sent_tokenize(block_text)
    groups = causal_group(sentences, cfg.causal_group_max_words)
    header = f"{header_title} · {section}".strip(" ·")

    # windowing
    if cfg.embed_token_limit and cfg.embed_token_limit > 0:
        token_windows = window_by_tokens(
            " ".join(groups), tok, cfg.embed_token_limit, cfg.overlap_ratio
        )
        pieces = [tw[0] for tw in token_windows]
    else:
        pieces = window_by_sentences(groups, cfg.max_sentences, cfg.sent_overlap)

    # enforce hard caps
    out_p = []
    if cfg.max_tokens_per_chunk > 0:
        for p in pieces:
            if tok.count(p) <= cfg.max_tokens_per_chunk:
                out_p.append(p)
            else:
                sents = sent_tokenize(p)
                packed = _pack_sentences_to_hard_cap(sents, tok, cfg)
                packed = _apply_adaptive_overlap(packed, tok, cfg, base_overlap_sent=1)
                out_p.extend(packed)
    else:
        out_p = pieces

    # filter too-short pieces
    min_words_eff = max(MIN_CHUNK_WORD_THRESHOLD, int(cfg.min_words * 0.8))
    filtered = [p for p in out_p if len(p.split()) >= min_words_eff]
    force_keep = False
    if not filtered and out_p:
        filtered = [max(out_p, key=lambda x: len(x.split()))]
        force_keep = True

    final_out = []
    for p in filtered:
        if tok.count(p) > cfg.max_tokens_per_chunk:
            sents = sent_tokenize(p)
            final_out.extend(_pack_sentences_to_hard_cap(sents, tok, cfg))
        else:
            final_out.append(p)
    filtered = final_out

    rows = []
    seq = article_seq_start
    images_json = json.dumps(images_for_block, ensure_ascii=False)

    for idx_in_block, body in enumerate(filtered):
        header_prefix = header + "\n\n" if header else ""
        cap = int(cfg.max_tokens_per_chunk or 0)

        emit_list = []
        if cap > 0:
            htok = tok.count(header_prefix)
            if htok >= cap:
                header_prefix = ""
                htok = 0
            body_allow = max(1, cap - htok)
            if tok.count(header_prefix + body) <= cap:
                emit_list = [body]
            else:
                cfg2 = Config(
                    embed_token_limit=cfg.embed_token_limit,
                    overlap_ratio=cfg.overlap_ratio,
                    max_sentences=cfg.max_sentences,
                    sent_overlap=cfg.sent_overlap,
                    min_words=cfg.min_words,
                    min_chunk_word_threshold=cfg.min_chunk_word_threshold,
                    causal_group_max_words=cfg.causal_group_max_words,
                    model_tokenizer=cfg.model_tokenizer,
                    max_tokens_per_chunk=body_allow,
                    adaptive_overlap_threshold=cfg.adaptive_overlap_threshold,
                    long_sent_token_threshold=cfg.long_sent_token_threshold,
                )
                emit_list = _pack_sentences_to_hard_cap(sent_tokenize(body), tok, cfg2)
                emit_list = _apply_adaptive_overlap(emit_list, tok, cfg2, base_overlap_sent=1)
        else:
            emit_list = [body]

        for sub_idx, piece in enumerate(emit_list):
            if not piece:
                continue
            if not force_keep and len(piece.split()) < min_words_eff:
                continue

            body_clean = fix_mojibake(piece)
            embed_text = fix_mojibake(f"{header_prefix}{body_clean}")
            text_with_imgs = (embed_text + (" " + " ".join(img_texts) if img_texts else "")).strip()

            if cap > 0 and tok.count(embed_text) > cap and tiktoken and tok.enc:
                encoded = tok.enc.encode(embed_text)
                embed_text = tok.enc.decode(encoded[:cap])

            final_token_len = tok.count(embed_text)
            chunk_id = sha256_hex(
                f"{article_id}|{section}|{idx_in_block}_{sub_idx}|{body_clean[:80]}"
            )
            rows.append({
                "article_id": article_id,
                "chunk_id": chunk_id,
                "title": header_title or None,
                "section": section,
                "chunk_idx": (
                    str(idx_in_block)
                    if len(emit_list) == 1
                    else f"{idx_in_block}.{sub_idx}"
                ),
                "article_seq": seq,
                "embed_text": embed_text,
                "body": body_clean,
                "text_with_images": text_with_imgs,
                "images": images_json,
                "source_relpath": source_relpath,
                "token_len": final_token_len,
                "word_len": len(body_clean.split()),
            })
            seq += 1

    return rows, seq

# ----------------------------- process_article -----------------------------------
def process_article(path: Path, site_root: Path, cfg: Config, tok: TokenCounter):
    soup = read_html(path)
    content_root = find_main_content(soup)
    meta = extract_meta(soup)

    relpath = str(path.relative_to(site_root)).replace("\\", "/")
    canonical_url = meta.get("url") or relpath
    article_id = sha256_hex(canonical_url)

    ctx = FlushContext(current_section="Main", article_seq=0)
    references = []
    in_references_section = False

    def flush_paragraph_buffer():
        if not ctx.paragraph_buffer:
            return
        combined = " ".join(ctx.paragraph_buffer).strip()
        combined = fix_mojibake(combined)

        # STRICT REFERENCE FILTER (REMOVE BLOCK)
        if IS_REFERENCE_BLOCK_RE.search(combined):
            ctx.paragraph_buffer = []
            ctx.images = []
            return

        combined = prune_trailing_biblio(combined, cfg.min_words)
        ctx.paragraph_buffer = []

        if len(combined.split()) < cfg.min_words:
            ctx.images = []
            return

        header_title = meta.get("title_og") or meta.get("title") or ""
        image_texts = list(dict.fromkeys([
            img.get("image_text", "").strip()
            for img in ctx.images if img.get("image_text")
        ]))

        rows, new_seq = _materialize_chunks_for_block(
            block_text=combined,
            header_title=header_title,
            section=ctx.current_section,
            cfg=cfg,
            tok=tok,
            img_texts=image_texts,
            images_for_block=ctx.images,
            article_id=article_id,
            article_seq_start=ctx.article_seq,
            source_relpath=relpath
        )

        if rows:
            ctx.output_chunks.extend(rows)
            ctx.article_seq = new_seq
        ctx.images = []

    # Traverse DOM
    for node in content_root.find_all(
        ["h1", "h2", "h3", "p", "blockquote", "ul", "ol", "figure"],
        recursive=True
    ):
        tag = node.name.lower()

        # Headings
        if tag in {"h1", "h2", "h3"}:
            heading = clean_text(node)
            flush_paragraph_buffer()
            in_references_section = bool(REFERENCE_RE.match(heading))
            if heading:
                ctx.current_section = heading
            continue

        # Reference section
        if in_references_section:
            txt = clean_text(node)
            if txt:
                references.append(txt)
            continue

        # Images
        if tag == "figure":
            ctx.images.extend(find_images(node))
            continue

        # Blockquotes — handle once, skip inner <p> duplication
        if tag == "blockquote":
            quote = strip_inline_citations(clean_text(node))
            if not quote or CAPTION_RE.search(quote):
                continue
            if IS_REFERENCE_BLOCK_RE.search(quote):
                continue
            quote = prune_trailing_biblio(quote, cfg.min_words)
            if not quote:
                continue

            # Make the quote its own block
            flush_paragraph_buffer()
            ctx.paragraph_buffer.append(quote)
            ctx.images.extend(find_images(node))
            flush_paragraph_buffer()
            continue

        # Paragraphs (but skip those inside blockquotes to avoid duplication)
        if tag == "p":
            if node.find_parent("blockquote") is not None:
                continue

            paragraph = strip_inline_citations(clean_text(node))
            if not paragraph or CAPTION_RE.search(paragraph):
                continue

            # STRICT REFERENCE FILTER
            if IS_REFERENCE_BLOCK_RE.search(paragraph):
                continue

            paragraph = prune_trailing_biblio(paragraph, cfg.min_words)
            if not paragraph:
                continue

            ctx.paragraph_buffer.append(paragraph)
            ctx.images.extend(find_images(node))
            continue

        # Lists
        if tag in {"ul", "ol"}:
            items = []
            for li in node.find_all("li", recursive=False):
                item = strip_inline_citations(clean_text(li))
                if not item or CAPTION_RE.search(item):
                    continue

                # STRICT REFERENCE FILTER
                if IS_REFERENCE_BLOCK_RE.search(item):
                    continue

                item = prune_trailing_biblio(item, cfg.min_words)
                if not item:
                    continue

                items.append(item)
                ctx.images.extend(find_images(li))
            if items:
                ctx.paragraph_buffer.append(" ".join(items))
            continue

    flush_paragraph_buffer()

    article_row = {
        "article_id": article_id,
        "title": meta.get("title_og") or meta.get("title"),
        "url": canonical_url,
        "published": meta.get("published"),
        "updated": meta.get("updated"),
        "source_relpath": relpath,
        "references_text": "\n".join(references) if references else None,
        "references_count": len(references),
    }
    return article_row, ctx.output_chunks

# ----------------------------- Sanity & dedup -----------------------------------
def sanity_check_mojibake(df: pd.DataFrame, col: str):
    if col not in df:
        return
    needles = ["â", "Â", "‚Ä", "�"]
    pattern = "|".join(map(re.escape, needles))
    mask = df[col].fillna("").astype(str).str.contains(pattern, na=False)
    if mask.any():
        log.warning("⚠️ %d rows with possible mojibake in %s", mask.sum(), col)
    else:
        log.info("No mojibake artifacts in %s", col)

def dedup_near_exact(df: pd.DataFrame, text_col="body") -> pd.DataFrame:
    norm = (
        df[text_col]
        .fillna("")
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    fp = norm.apply(sha256_hex)
    before = len(df)
    dedup = df.loc[~fp.duplicated()].copy()
    log.info("De-duplicated chunks: %d → %d", before, len(dedup))
    return dedup

# ----------------------------- Save outputs -------------------------------------
def build_and_save(articles, chunks, outdir: Path, no_overwrite: bool):
    if not chunks:
        log.error("No chunks produced.")
        return

    df_articles = pd.DataFrame(articles)
    df_chunks = pd.DataFrame(chunks)

    sanity_check_mojibake(df_chunks, "embed_text")
    df_chunks = dedup_near_exact(df_chunks, "body")

    sort_keys = ["source_relpath", "article_seq" if "article_seq" in df_chunks.columns else "chunk_idx"]
    df_chunks = df_chunks.sort_values(sort_keys).reset_index(drop=True)
    df_articles = df_articles.sort_values("source_relpath").reset_index(drop=True)

    outdir.mkdir(parents=True, exist_ok=True)
    articles_path = outdir / "all_articles.jsonl"
    chunks_parquet = outdir / "chunks.parquet"

    if (articles_path.exists() or chunks_parquet.exists()) and no_overwrite:
        log.error("Output exists; rerun without --no-overwrite.")
        return

    df_articles.to_json(articles_path, orient="records", lines=True, force_ascii=False)

    try:
        df_chunks.to_parquet(chunks_parquet, index=False)
        log.info("Wrote %s", chunks_parquet)
    except Exception as ex:
        fallback = outdir / "chunks.jsonl"
        log.warning("Parquet write failed (%s). Falling back to JSONL", ex.__class__.__name__)
        df_chunks.to_json(fallback, orient="records", lines=True, force_ascii=False)

    avg_words = df_chunks["word_len"].mean() if "word_len" in df_chunks else 0
    avg_tokens = df_chunks["token_len"].mean() if "token_len" in df_chunks else 0
    max_tokens = df_chunks["token_len"].max() if "token_len" in df_chunks else 0
    max_words = df_chunks["word_len"].max() if "word_len" in df_chunks else 0
    min_tokens = df_chunks["token_len"].min() if "token_len" in df_chunks else 0
    min_words = df_chunks["word_len"].min() if "word_len" in df_chunks else 0

    log.info("Saved %d articles, %d chunks", len(df_articles), len(df_chunks))
    log.info("Avg words/chunk: %.1f | Avg tokens/chunk: %.1f", avg_words, avg_tokens)
    log.info("Max words: %d | Max tokens: %d", max_words, max_tokens)
    log.info("Min words: %d | Min tokens: %d", min_words, min_tokens)
    log.info("%s", articles_path)

# ----------------------------- Main ----------------------------------------------
def main(cli_args: argparse.Namespace):
    cfg = Config(
        embed_token_limit=cli_args.embed_token_limit,
        overlap_ratio=cli_args.overlap_ratio,
        max_sentences=cli_args.max_sentences,
        sent_overlap=cli_args.sent_overlap,
        min_words=cli_args.min_words,
        min_chunk_word_threshold=MIN_CHUNK_WORD_THRESHOLD,
        causal_group_max_words=CAUSAL_GROUP_MAX_WORDS,
        model_tokenizer=cli_args.model_tokenizer,
        max_tokens_per_chunk=cli_args.max_tokens_per_chunk,
        adaptive_overlap_threshold=cli_args.adaptive_overlap_threshold,
        long_sent_token_threshold=cli_args.long_sent_token_threshold,
    )

    site_root = detect_site_root(cli_args.root.resolve())
    index_files = list_index_files(site_root)
    if not index_files:
        log.error("No HTML files found under %s", cli_args.root)
        return

    if cli_args.debug:
        log.setLevel(logging.DEBUG)

    outdir = cli_args.outdir.resolve()
    tok = TokenCounter(cfg)

    articles = []
    chunks = []

    if cli_args.workers and cli_args.workers > 1:

        with ThreadPoolExecutor(max_workers=cli_args.workers) as pool:
            futures = [
                pool.submit(process_article, p, site_root, cfg, tok)
                for p in index_files
            ]

            if tqdm is not None and not cli_args.no_tqdm:
                iterator = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Parallel articles"
                )
            else:
                iterator = as_completed(futures)

            for fut in iterator:
                try:
                    art, ch = fut.result()
                except Exception as e:
                    log.exception("Worker crash: %s", e)
                    continue

                if art and ch:
                    articles.append(art)
                    chunks.extend(ch)
    else:
        iterator = index_files
        if tqdm is not None and not cli_args.no_tqdm:
            iterator = tqdm(index_files, desc="Processing articles")
        for path_ in iterator:
            try:
                art, ch = process_article(path_, site_root, cfg, tok)
            except Exception as e:
                log.exception("Failed: %s (%s)", path_, e)
                continue
            if art and ch:
                articles.append(art)
                chunks.extend(ch)

    build_and_save(articles, chunks, outdir, cli_args.no_overwrite)

# ----------------------------- CLI ----------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSK Neurology mirror → extracted metadata + chunked text")

    parser.add_argument("--root", type=Path, default=Path("MSKArticlesINDEX"),
                        help="Root folder of HTTrack mirror")
    parser.add_argument("--outdir", type=Path, default=Path("MSKArticlesINDEX"),
                        help="Directory to write output files")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="Do not overwrite existing outputs")

    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable progress bars")

    # Sentence mode
    parser.add_argument("--max-sentences", type=int, default=DEFAULT_MAX_SENTENCES)
    parser.add_argument("--sent-overlap", type=int, default=DEFAULT_SENT_OVERLAP)
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)

    # Token windowing
    parser.add_argument("--embed-token-limit", type=int, default=DEFAULT_EMBED_TOKEN_LIMIT)
    parser.add_argument("--overlap-ratio", type=float, default=DEFAULT_OVERLAP_RATIO)
    parser.add_argument("--model-tokenizer", type=str, default=DEFAULT_MODEL_TOKENIZER)

    # Hard cap in sentence mode
    parser.add_argument("--max-tokens-per-chunk", type=int, default=DEFAULT_MAX_TOKENS_PER_CHUNK)
    parser.add_argument("--adaptive-overlap-threshold", type=int, default=DEFAULT_ADAPTIVE_OVERLAP_THRESH)
    parser.add_argument("--long-sent-token-threshold", type=int, default=DEFAULT_LONG_SENT_TOKEN_THRESH)

    # Parallelism
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)

    args_ns = parser.parse_args()
    main(args_ns)
