#!/usr/bin/env python3
"""
resolve_citations.py — Step 2 of the cited-literature backfill.

Consumes the manifest from harvest_citations.py and resolves the DETERMINISTIC
slice — bare DOIs, PMIDs, and PMC IDs — into real work records via the OpenAlex
API. These identifiers resolve with confidence 1.0 (no fuzzy matching).

The fuzzy author-year slice (~636 strings) is intentionally NOT handled here;
it needs a separate scored resolver. Author-year strings that OpenAlex cannot
confirm belong in a manual-review bucket, and a safety-sensitive corpus must not
silently ingest a mis-matched paper.

Outputs (default under MSKArticlesINDEX/):
  resolved_deterministic.jsonl   one record per uniquely resolved work
  resolve_failures.jsonl         identifiers OpenAlex could not resolve

Network: OpenAlex only (free, no key). Uses the polite pool via ?mailto and
caches raw responses on disk so reruns are free/idempotent.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

OPENALEX = "https://api.openalex.org/works"
MAILTO = "zdraconborn@gmail.com"          # OpenAlex polite-pool etiquette
SLEEP_S = 0.12                            # < 10 req/s
CACHE_DIR = Path(".cache_openalex")


def _cache_path(base: Path, kind: str, ident: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", ident)
    return base / CACHE_DIR / f"{kind}_{safe}.json"


def _get_json(url: str) -> Optional[Any]:
    req = urllib.request.Request(url, headers={"User-Agent": f"msk-backfill ({MAILTO})"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    finally:
        time.sleep(SLEEP_S)


def normalize_doi(doi: str) -> str:
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.I)
    # regex over prose can grab trailing punctuation; peel it off
    return doi.rstrip(").,;:'\"]")


def fetch_work(kind: str, ident: str, outdir: Path) -> Optional[Dict[str, Any]]:
    """kind in {doi, pmid, pmcid}. Returns raw OpenAlex work dict or None."""
    cp = _cache_path(outdir, kind, ident)
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8")) or None

    q = urllib.parse.quote(ident, safe="")
    if kind == "doi":
        url = f"{OPENALEX}/doi:{q}?mailto={MAILTO}"
        data = _get_json(url)
    elif kind == "pmid":
        url = f"{OPENALEX}/pmid:{q}?mailto={MAILTO}"
        data = _get_json(url)
    elif kind == "pmcid":
        # no direct /pmcid: route — filter instead, take the single hit
        url = f"{OPENALEX}?filter=ids.pmcid:{q}&per_page=1&mailto={MAILTO}"
        page = _get_json(url)
        results = (page or {}).get("results") or []
        data = results[0] if results else None
    else:
        raise ValueError(kind)

    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(data or {}, ensure_ascii=False), encoding="utf-8")
    return data


def reconstruct_abstract(inv: Optional[Dict[str, List[int]]]) -> Optional[str]:
    if not inv:
        return None
    positions: List[Tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions) or None


def flatten(work: Dict[str, Any]) -> Dict[str, Any]:
    ids = work.get("ids") or {}
    prim = (work.get("primary_location") or {}).get("source") or {}
    oa = work.get("open_access") or {}
    authors = [
        (a.get("author") or {}).get("display_name")
        for a in (work.get("authorships") or [])
    ]
    return {
        "openalex_id": work.get("id"),
        "doi": (work.get("doi") or "").replace("https://doi.org/", "") or None,
        "pmid": (ids.get("pmid") or "").rsplit("/", 1)[-1] or None,
        "pmcid": (ids.get("pmcid") or "").rsplit("/", 1)[-1] or None,
        "title": work.get("title") or work.get("display_name"),
        "authors": [a for a in authors if a],
        "year": work.get("publication_year"),
        "venue": prim.get("display_name"),
        "type": work.get("type"),
        "cited_by_count": work.get("cited_by_count"),
        "oa_status": oa.get("oa_status"),
        "oa_url": oa.get("oa_url"),
        "is_oa": oa.get("is_oa"),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "topics": [t.get("display_name") for t in (work.get("topics") or [])[:3]],
    }


def collect_identifiers(recs: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[str]]:
    """Map (kind, ident) -> list of source_relpaths citing it. DOIs also mined
    from scholarly_link hrefs that point at doi.org."""
    idents: Dict[Tuple[str, str], List[str]] = {}

    def add(kind: str, ident: str, src: str):
        if not ident:
            return
        key = (kind, ident)
        idents.setdefault(key, [])
        if src not in idents[key]:
            idents[key].append(src)

    doi_in_url = re.compile(r"doi\.org/(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.I)
    for r in recs:
        src = r.get("source_relpath")
        for d in r.get("dois", []):
            add("doi", normalize_doi(d), src)
        for p in r.get("pmids", []):
            add("pmid", str(p), src)
        for pmc in r.get("pmc_ids", []):
            add("pmcid", pmc.upper(), src)
        for link in r.get("scholarly_links", []):
            m = doi_in_url.search(link.get("href", ""))
            if m:
                add("doi", normalize_doi(m.group(1)), src)
    return idents


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("MSKArticlesINDEX/citations.jsonl"))
    ap.add_argument("--outdir", type=Path, default=Path("MSKArticlesINDEX"))
    ap.add_argument("--limit", type=int, default=0, help="Resolve at most N identifiers (0 = all)")
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.input.open(encoding="utf-8")]
    idents = collect_identifiers(recs)

    # Prefer DOI > PMID > PMC when the same work appears under multiple ids we
    # can't yet know are the same; we still resolve each id but dedup outputs by
    # the OpenAlex id we get back.
    order = {"doi": 0, "pmid": 1, "pmcid": 2}
    items = sorted(idents.items(), key=lambda kv: order[kv[0][0]])
    if args.limit:
        items = items[: args.limit]

    print(f"Deterministic identifiers to resolve: {len(items)} "
          f"({sum(1 for (k,_),_ in items if k=='doi')} DOI, "
          f"{sum(1 for (k,_),_ in items if k=='pmid')} PMID, "
          f"{sum(1 for (k,_),_ in items if k=='pmcid')} PMC)")

    resolved: Dict[str, Dict[str, Any]] = {}   # openalex_id -> record
    failures: List[Dict[str, Any]] = []
    done = 0
    for (kind, ident), sources in items:
        try:
            work = fetch_work(kind, ident, args.outdir)
        except Exception as e:
            failures.append({"kind": kind, "identifier": ident, "error": f"{type(e).__name__}: {e}", "cited_by": sources})
            continue
        if not work:
            failures.append({"kind": kind, "identifier": ident, "error": "not_found", "cited_by": sources})
            continue
        flat = flatten(work)
        oid = flat["openalex_id"] or f"{kind}:{ident}"
        rec = resolved.get(oid)
        if rec is None:
            flat.update({
                "confidence": 1.0,
                "resolved_via": [f"{kind}:{ident}"],
                "cited_by": list(sources),
                "source_type": "literature",
            })
            resolved[oid] = flat
        else:
            rec["resolved_via"].append(f"{kind}:{ident}")
            for s in sources:
                if s not in rec["cited_by"]:
                    rec["cited_by"].append(s)
        done += 1
        if done % 25 == 0:
            print(f"  ...{done}/{len(items)} resolved")

    args.outdir.mkdir(parents=True, exist_ok=True)
    rp = args.outdir / "resolved_deterministic.jsonl"
    fp = args.outdir / "resolve_failures.jsonl"
    with rp.open("w", encoding="utf-8") as fh:
        for rec in sorted(resolved.values(), key=lambda r: -(r.get("cited_by_count") or 0)):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with fp.open("w", encoding="utf-8") as fh:
        for f in failures:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")

    n_oa = sum(1 for r in resolved.values() if r.get("is_oa"))
    n_abs = sum(1 for r in resolved.values() if r.get("abstract"))
    print()
    print(f"Resolved unique works : {len(resolved)}")
    print(f"  with abstract       : {n_abs}")
    print(f"  open-access full text: {n_oa}")
    print(f"Failures (unresolved) : {len(failures)}")
    print(f"Wrote {rp}")
    print(f"Wrote {fp}")


if __name__ == "__main__":
    main()
