#!/usr/bin/env python3
"""
fetch_literature.py — Step 3 of the cited-literature backfill.

Consumes resolved_deterministic.jsonl and assembles an ingest-ready text payload
per work, choosing the best AVAILABLE text without fragile publisher scraping:

  1. oa_fulltext  — legal open-access HTML from PMC (NCBI), extracted with bs4
  2. abstract     — OpenAlex-reconstructed abstract (fallback, always if present)
  3. none         — neither available (metadata-only record)

Deliberately dependency-light: PMC HTML only (bs4). Publisher PDFs
(Wiley/BMJ/Springer/Thieme/Hindawi) are left as a follow-up that needs a PDF
library (pypdf) — those works fall back to their abstract here.

Does NOT assign evidence_tier and does NOT write to chroma_store/. This is the
last step before the protected citation-grounding surface; the ingest step
(chunk -> embed -> index) is intentionally separate.

Output (default under MSKArticlesINDEX/):
  literature_corpus.jsonl   ingest-ready {metadata, text, text_source}
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

MAILTO = "zdraconborn@gmail.com"
SLEEP_S = 0.34                          # be gentle with NCBI (< 3 req/s)
CACHE_DIR = Path(".cache_fulltext")
HEADERS = {"User-Agent": f"msk-backfill/1.0 (mailto:{MAILTO})"}

# PMC article containers, most-specific first.
PMC_CONTENT_SELECTORS = [
    ("div", {"class": "jig-ncbiinpagenav"}),
    ("div", {"class": "article"}),
    ("article", {}),
    ("div", {"id": "maincontent"}),
]
# Sections to drop from full text (refs/acks/footnotes/nav).
DROP_SECTION_RE = re.compile(r"(reference|acknowledg|footnote|author information|"
                             r"conflict|funding|supplementary)", re.I)


def is_pmc(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return "ncbi.nlm.nih.gov" in host and "/pmc/" in url.lower()


def cache_path(base: Path, key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)[:120]
    return base / CACHE_DIR / f"{safe}.html"


def fetch_html(url: str, base: Path) -> Optional[str]:
    cp = cache_path(base, url)
    if cp.exists():
        return cp.read_text(encoding="utf-8") or None
    try:
        r = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        time.sleep(SLEEP_S)
        html = r.text if r.status_code == 200 else ""
    except Exception:
        html = ""
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(html, encoding="utf-8")
    return html or None


def extract_pmc_text(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    root = None
    for name, attrs in PMC_CONTENT_SELECTORS:
        el = soup.find(name, attrs=attrs)
        if el:
            root = el
            break
    if root is None:
        return None

    # Drop reference/ack/footnote sections by their heading text.
    for sec in root.find_all(["section", "div"]):
        h = sec.find(["h2", "h3"])
        if h and DROP_SECTION_RE.search(h.get_text(" ", strip=True)):
            sec.decompose()
    for bad in root.find_all(["nav", "table", "figure", "script", "style"]):
        bad.decompose()

    paras = [p.get_text(" ", strip=True) for p in root.find_all("p")]
    paras = [re.sub(r"\s+", " ", p) for p in paras if len(p.split()) >= 8]
    text = "\n\n".join(paras).strip()
    return text or None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=Path("MSKArticlesINDEX/resolved_deterministic.jsonl"))
    ap.add_argument("--outdir", type=Path, default=Path("MSKArticlesINDEX"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-fulltext-chars", type=int, default=800,
                    help="Below this, discard scraped full text and fall back to abstract")
    args = ap.parse_args()

    works = [json.loads(l) for l in args.input.open(encoding="utf-8")]
    if args.limit:
        works = works[: args.limit]

    out: List[Dict[str, Any]] = []
    n_full = n_abs = n_none = 0
    for i, w in enumerate(works, 1):
        text: Optional[str] = None
        source = "none"
        oa_url = w.get("oa_url")

        if oa_url and is_pmc(oa_url):
            html = fetch_html(oa_url, args.outdir)
            if html:
                ft = extract_pmc_text(html)
                if ft and len(ft) >= args.min_fulltext_chars:
                    text, source = ft, "oa_fulltext"

        if text is None and w.get("abstract"):
            text, source = w["abstract"], "abstract"

        if source == "oa_fulltext":
            n_full += 1
        elif source == "abstract":
            n_abs += 1
        else:
            n_none += 1

        out.append({
            "openalex_id": w.get("openalex_id"),
            "doi": w.get("doi"),
            "pmid": w.get("pmid"),
            "pmcid": w.get("pmcid"),
            "title": w.get("title"),
            "authors": w.get("authors"),
            "year": w.get("year"),
            "venue": w.get("venue"),
            "topics": w.get("topics"),
            "oa_status": w.get("oa_status"),
            "oa_url": oa_url,
            "cited_by": w.get("cited_by"),
            "source_type": "literature",
            "text_source": source,
            "text_chars": len(text) if text else 0,
            "text": text,
        })
        if i % 25 == 0:
            print(f"  ...{i}/{len(works)}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    op = args.outdir / "literature_corpus.jsonl"
    with op.open("w", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print(f"Works processed        : {len(out)}")
    print(f"  oa_fulltext (PMC)    : {n_full}")
    print(f"  abstract fallback    : {n_abs}")
    print(f"  metadata-only (none) : {n_none}")
    print(f"Wrote {op}")


if __name__ == "__main__":
    main()
