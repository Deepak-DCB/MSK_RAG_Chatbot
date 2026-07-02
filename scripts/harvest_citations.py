#!/usr/bin/env python3
"""
harvest_citations.py — read-only citation harvester for the MSK corpus.

Re-parses the raw HTML mirror (MSKArticlesINDEX/mskneurology.com/) BEFORE the
extractor strips citations, and pulls out every inline author-year citation and
every outbound scholarly link/identifier per source article.

This is the first step of the "backfill cited literature" pipeline. It writes
nothing into the live retrieval path — it only emits a citations manifest that
the (future) resolver will consume.

Outputs (default under MSKArticlesINDEX/):
  citations.jsonl   one record per source article with its harvested citations
  citations_unique.jsonl  deduped unique cited works (author-year key + links)

Zero-cost: no network, no API keys, no writes to chroma_store/.

Usage:
  python scripts/harvest_citations.py                 # harvest + summary
  python scripts/harvest_citations.py --summary-only   # counts only, no write
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import unicodedata
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup, Tag

warnings.filterwarnings("ignore")  # silence XMLParsedAsHTMLWarning on feed files

# Reuse the extractor's content-selector priority so we read article body only,
# not nav/footer/sidebar noise.
CONTENT_SELECTORS = [
    ("div", {"class": "entry-content"}),
    ("article", {}),
    ("main", {}),
    ("div", {"id": "content"}),
]

# Domains that indicate a scholarly reference target.
SCHOLARLY_HOST_HINTS = (
    "pubmed", "ncbi.nlm.nih.gov", "/pmc/", "doi.org", "sciencedirect",
    "springer", "link.springer", "wiley", "onlinelibrary", "tandfonline",
    "sagepub", "jamanetwork", "bmj.com", "thelancet", "nejm.org",
    "researchgate", "scholar.google", "semanticscholar", "europepmc",
    "oup.com", "academic.oup", "karger", "frontiersin", "mdpi.com",
    "physoc.onlinelibrary", "journals.lww", "jospt.org", "jbjs.org",
)

# Months and non-author leading words that produce false author-year hits
# (e.g. "May 2024", "Epub 2007", "Edition 2014", "Dec 2019").
_STOP_LEADING = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec",
    "epub", "edition", "vol", "volume", "no", "pp", "page", "pages", "figure",
    "table", "chapter", "in", "the", "since", "during", "by", "copyright",
    "published", "updated", "accessed", "retrieved", "version",
}

# Author-year citation:
#   Surname et al., 2016
#   Surname & Surname, 2007
#   Surname and Surname 2015
#   Surname, 2006  |  Surname 2015
# Leading surname allows Latin-1 diacritics common in author names.
_NAME = r"[A-ZÅÄÖØÆËÉÜ][A-Za-zÅÄÖØÆËÉÜáàâäãåçéèêëíìîïñóòôöõúùûüÿćčžšß'\-]+"
AUTHOR_YEAR_RE = re.compile(
    rf"""
    \b
    ({_NAME})                                   # first surname  (grp 1)
    (                                           # optional connector (grp 2)
        \s+et\ al\.?
        | \s*(?:&|and)\s*{_NAME}
    )?
    ,?\s*
    ((?:19|20)\d{{2}}[a-z]?)                     # year, optional a/b suffix (grp 3)
    \b
    """,
    re.VERBOSE,
)

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.I)
PMID_RE = re.compile(r"\bPMID\s*[:=]?\s*(\d{5,9})\b", re.I)
PMC_RE = re.compile(r"\bPMC\d{5,9}\b", re.I)


def find_main_content(soup: BeautifulSoup) -> Tag:
    for name, attrs in CONTENT_SELECTORS:
        el = soup.find(name, attrs=attrs)
        if el:
            return el
    return soup.body or soup


def is_real_article(path: Path) -> bool:
    """Skip HTTrack RSS/feed mirrors and comment feeds — body articles only."""
    p = str(path).replace("\\", "/").lower()
    if "/feed/" in p or "/comment" in p or "/page/" in p:
        return False
    return path.name.lower().startswith("index")


def clean_link_text(a: Tag) -> str:
    return re.sub(r"\s+", " ", a.get_text(" ", strip=True)).strip()


def normalize_key(surname: str, connector: str, year: str) -> str:
    """Dedup key: lowercase surname + 'etal' flag + year. Loses second author
    on purpose so 'Smith & Jones 2010' and 'Smith 2010' don't over-split; the
    resolver refines this later."""
    etal = "+" if connector and "et al" in connector.lower() else ""
    return f"{surname.lower()}{etal}|{year}"


def harvest_article(path: Path, site_root: Path) -> Optional[Dict[str, Any]]:
    raw = path.read_bytes().decode("utf-8", errors="replace")
    raw = unicodedata.normalize("NFKC", raw)
    try:
        soup = BeautifulSoup(raw, "lxml")
    except Exception:
        soup = BeautifulSoup(raw, "html.parser")

    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else (
        soup.title.get_text(strip=True) if soup.title else None)
    url_meta = soup.find("meta", attrs={"property": "og:url"})
    url = url_meta["content"].strip() if url_meta and url_meta.get("content") else None

    content = find_main_content(soup)
    text = content.get_text(" ", strip=True)

    # --- author-year citations ---
    author_year: List[Dict[str, str]] = []
    seen_local = set()
    for m in AUTHOR_YEAR_RE.finditer(text):
        surname, connector, year = m.group(1), (m.group(2) or ""), m.group(3)
        if surname.lower() in _STOP_LEADING:
            continue
        raw_cite = re.sub(r"\s+", " ", m.group(0)).strip()
        key = normalize_key(surname, connector, year)
        if key in seen_local:
            continue
        seen_local.add(key)
        author_year.append({
            "raw": raw_cite,
            "surname": surname,
            "connector": connector.strip(),
            "year": year,
            "key": key,
        })

    # --- scholarly links ---
    links: List[Dict[str, str]] = []
    for a in content.find_all("a", href=True):
        href = a["href"].strip()
        if any(h in href.lower() for h in SCHOLARLY_HOST_HINTS):
            links.append({"href": href, "text": clean_link_text(a)})

    # --- bare identifiers in text ---
    dois = sorted(set(DOI_RE.findall(text)))
    pmids = sorted(set(PMID_RE.findall(text)))
    pmcs = sorted(set(PMC_RE.findall(text)))

    relpath = str(path.relative_to(site_root)).replace("\\", "/")
    return {
        "source_article": title,
        "source_url": url,
        "source_relpath": relpath,
        "author_year_count": len(author_year),
        "scholarly_link_count": len(links),
        "author_year": author_year,
        "scholarly_links": links,
        "dois": dois,
        "pmids": pmids,
        "pmc_ids": pmcs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mirror", type=Path,
                    default=Path("MSKArticlesINDEX/mskneurology.com"),
                    help="Root of the HTTrack HTML mirror")
    ap.add_argument("--outdir", type=Path, default=Path("MSKArticlesINDEX"))
    ap.add_argument("--summary-only", action="store_true",
                    help="Print counts, do not write output files")
    args = ap.parse_args()

    site_root = args.mirror.resolve()
    if not site_root.exists():
        raise SystemExit(f"Mirror not found: {site_root}")

    files = [Path(p) for p in glob.glob(str(site_root / "**" / "index*"), recursive=True)]
    articles = sorted({p for p in files if p.is_file() and is_real_article(p)})

    records: List[Dict[str, Any]] = []
    for path in articles:
        rec = harvest_article(path, site_root)
        if rec and (rec["author_year_count"] or rec["scholarly_link_count"]):
            records.append(rec)

    # --- global dedup of unique cited works ---
    unique: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        for cite in rec["author_year"]:
            u = unique.setdefault(cite["key"], {
                "key": cite["key"],
                "surname": cite["surname"],
                "year": cite["year"],
                "sample_raw": cite["raw"],
                "cited_by": [],
            })
            if rec["source_relpath"] not in u["cited_by"]:
                u["cited_by"].append(rec["source_relpath"])

    # link/identifier tallies
    all_links = Counter()
    for rec in records:
        for l in rec["scholarly_links"]:
            all_links[l["href"]] += 1
    all_dois = {d for rec in records for d in rec["dois"]}

    total_ay = sum(r["author_year_count"] for r in records)
    total_links = sum(r["scholarly_link_count"] for r in records)

    print(f"Articles with citations : {len(records)} / {len(articles)} scanned")
    print(f"Author-year citations   : {total_ay} raw, {len(unique)} unique works")
    print(f"Scholarly links         : {total_links} raw, {len(all_links)} unique URLs")
    print(f"Bare DOIs in text        : {len(all_dois)}")
    print()
    print("Top-cited authors (by # of source articles citing them):")
    top = sorted(unique.values(), key=lambda u: (-len(u["cited_by"]), u["surname"]))
    for u in top[:15]:
        print(f"  {u['sample_raw']:<28} cited in {len(u['cited_by'])} article(s)")

    if args.summary_only:
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    per_article = args.outdir / "citations.jsonl"
    uniq_path = args.outdir / "citations_unique.jsonl"
    with per_article.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with uniq_path.open("w", encoding="utf-8") as fh:
        for u in sorted(unique.values(), key=lambda x: (-len(x["cited_by"]), x["surname"])):
            fh.write(json.dumps(u, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote {per_article}")
    print(f"Wrote {uniq_path}")


if __name__ == "__main__":
    main()
