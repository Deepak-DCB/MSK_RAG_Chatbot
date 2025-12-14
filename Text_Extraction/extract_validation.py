#!/usr/bin/env python3
"""
extract_validation.py  —  MSK Neurology Extraction Validator (v8.5.1+)

Purpose
--------
Validate the outputs of `textExtract.py` to ensure the data is healthy, consistent,
and ready for downstream embedding or retrieval.

Checks performed
----------------
1. Article metadata integrity  →  correct fields, unique IDs
2. Chunk file integrity         →  correct schema, duplicate/orphan checks
3. Text encoding sanity         →  detect stray mojibake characters
4. Length statistics            →  average, min, and max chunk sizes

Usage
-----
    python extract_validation.py
    (Optional: update ROOT below if your MSKArticlesINDEX folder lives elsewhere.)

Interpretation
--------------
✅ = all good
⚠️ = mild warning (e.g., mojibake text)
❌ = something’s structurally wrong
"""

import json
import re
import sys
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(r"C:\Users\Draco\OneDrive\Documents\MSK_Triage_Chatbot\MSK_Chat\MSKArticlesINDEX")
ARTICLES_PATH = ROOT / "all_articles.jsonl"
CHUNKS_PATH = ROOT / "chunks.parquet"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path):
    """Load JSONL safely and report any malformed lines."""
    if not path.exists():
        print(f"❌ Missing file: {path}")
        sys.exit(1)

    articles = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            try:
                articles.append(json.loads(line))
            except Exception as e:
                print(f"⚠️  Could not parse line {i}: {e}")
    return articles


def check_article_schema(articles):
    """Confirm each article has the expected v8.5.1 fields."""
    required_fields = {
        "article_id", "title", "url",
        "published", "updated",
        "source_relpath", "references_count",
    }

    bad = []
    for a in articles:
        missing = required_fields - set(a.keys())
        if missing:
            bad.append((a.get("article_id", "<no-id>"), list(missing)))

    if bad:
        print(f"⚠️  {len(bad)} article(s) missing fields. Example:", bad[:3])
    else:
        print("✅ Article schema looks correct.")


def check_duplicate_ids(articles):
    """Detect duplicate article_ids."""
    ids = [a.get("article_id") for a in articles]
    dups = [x for x in set(ids) if ids.count(x) > 1]
    if dups:
        print(f"⚠️  Duplicate article_ids detected: {len(dups)} examples {dups[:3]}")
    else:
        print("✅ No duplicate article_ids found.")


def detect_mojibake(df, column: str):
    """Return count of rows with common mojibake characters."""
    weird = ["â", "√", "Â", "‚Ä", "�"]
    pattern = "|".join(map(re.escape, weird))
    mask = df[column].fillna("").astype(str).str.contains(pattern, regex=True)
    return int(mask.sum())


def summarize_lengths(df):
    """Print basic descriptive stats for chunk sizes."""
    if {"word_len", "token_len"} <= set(df.columns):
        avg_w = df["word_len"].mean()
        avg_t = df["token_len"].mean()
        min_w = df["word_len"].min()
        max_w = df["word_len"].max()
        print(f"🧮 Avg words/chunk: {avg_w:.1f} | Avg tokens/chunk: {avg_t:.1f}")
        print(f"↳ Range: {min_w}–{max_w} words")
    else:
        print("⚠️  Missing word_len/token_len columns.")


def preview(title: str, data):
    """Pretty-print a short preview of a dict or DataFrame row."""
    print(f"\n=== {title} ===")
    if isinstance(data, dict):
        print(json.dumps(data, indent=2)[:1000])
    else:
        print(data.to_dict())


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------

def main():
    print(f"📂 Validating extraction at: {ROOT}\n")

    # -------------------- Articles --------------------
    articles = load_jsonl(ARTICLES_PATH)
    print(f"Articles loaded: {len(articles)}")
    check_article_schema(articles)
    check_duplicate_ids(articles)

    # -------------------- Chunks ----------------------
    if not CHUNKS_PATH.exists():
        print(f"❌ Missing chunks.parquet at {CHUNKS_PATH}")
        sys.exit(1)

    df = pd.read_parquet(CHUNKS_PATH)
    print(f"Chunks loaded: {len(df)} rows")

    expected_cols = {
        "article_id", "chunk_id", "title", "section",
        "chunk_idx", "article_seq", "embed_text",
        "body", "text_with_images", "images",
        "source_relpath", "token_len", "word_len",
    }
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        print(f"⚠️  Missing expected columns: {missing_cols}")
    else:
        print("✅ Chunk schema looks correct.")

    # Duplicate chunk_ids
    dup_count = df["chunk_id"].duplicated().sum()
    print(f"Duplicate chunk_ids: {dup_count}")

    # Orphan check
    article_ids = {a["article_id"] for a in articles}
    orphan = df[~df["article_id"].isin(article_ids)]
    print(f"Orphan chunks (article_id not found): {len(orphan)}")

    # Mojibake scan
    bad_rows = detect_mojibake(df, "embed_text")
    print(f"Mojibake-affected rows: {bad_rows}")

    # Chunk size summary
    summarize_lengths(df)

    # -------------------- Summary ---------------------
    any_issues = dup_count > 0 or len(orphan) > 0 or bad_rows > 0 or missing_cols
    if any_issues:
        print("\n⚠️  Validation found minor issues (see above).")
    else:
        print("\n✅ Validation passed: dataset looks consistent and healthy.")

    # -------------------- Previews ---------------------
    if articles:
        preview("SAMPLE ARTICLE", articles[0])
    if not df.empty:
        preview("SAMPLE CHUNK", df.iloc[0])


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
