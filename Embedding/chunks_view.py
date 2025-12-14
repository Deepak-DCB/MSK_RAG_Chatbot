#!/usr/bin/env python3
"""
embedding_diff.py

Compare two embedding matrices (plain vs with-images), compute per-row similarity,
and export both the comparison summary and raw vectors as CSVs.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

emb_path = Path(r'C:\Users\Draco\OneDrive\Documents\MSK_Chat\Embedding\embeddings.npy')
emb_img_path = Path(r'C:\Users\Draco\OneDrive\Documents\MSK_Chat\Embedding\embeddingsImages.npy')
chunks_path = Path(r'C:\Users\Draco\OneDrive\Documents\MSK_Chat\MSKArticlesINDEX\chunks.parquet')

# ---------------------------------------------------------------------------
# Load embeddings
# ---------------------------------------------------------------------------

E = np.load(emb_path)
EI = np.load(emb_img_path)

if chunks_path.exists():
    df_chunks = pd.read_parquet(chunks_path)
else:
    df_chunks = pd.DataFrame({"row_idx": np.arange(min(E.shape[0], EI.shape[0]))})

# ---------------------------------------------------------------------------
# Align lengths
# ---------------------------------------------------------------------------

# If counts differ, trim to the smaller one
n = min(E.shape[0], EI.shape[0])
if E.shape[0] != EI.shape[0]:
    print(f"⚠️  Row count mismatch: plain={E.shape[0]}, with_images={EI.shape[0]} → truncating to {n}")
E = E[:n]
EI = EI[:n]

# Also trim the metadata
df_chunks = df_chunks.iloc[:n].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Cosine similarity and L2 distance
# ---------------------------------------------------------------------------

def safe_norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return n

En = E / safe_norm(E)
EIn = EI / safe_norm(EI)

cos_sim = (En * EIn).sum(axis=1)
l2_dist = np.linalg.norm(En - EIn, axis=1)

# ---------------------------------------------------------------------------
# Extract image/caption info if present
# ---------------------------------------------------------------------------

def parse_has_image(s):
    try:
        lst = json.loads(s) if isinstance(s, str) else (s or [])
        return len(lst) > 0
    except Exception:
        return False

def parse_cap_len(s):
    try:
        lst = json.loads(s) if isinstance(s, str) else (s or [])
        if not lst:
            return 0
        t = lst[0].get("image_text", "") if isinstance(lst[0], dict) else ""
        return len(str(t))
    except Exception:
        return 0

if "images" in df_chunks.columns:
    has_image = df_chunks["images"].apply(parse_has_image)
    cap_len = df_chunks["images"].apply(parse_cap_len)
else:
    has_image = pd.Series([False] * len(df_chunks))
    cap_len = pd.Series([0] * len(df_chunks))

# ---------------------------------------------------------------------------
# Build summary
# ---------------------------------------------------------------------------

summary_cols = [c for c in ["row_idx", "chunk_id", "article_id", "section"] if c in df_chunks.columns]

summary = pd.DataFrame({
    **({c: df_chunks[c] for c in summary_cols}),
    "has_image": has_image,
    "caption_text_len": cap_len,
    "cos_sim": cos_sim,
    "l2_dist": l2_dist,
})

summary_sorted = summary.sort_values("cos_sim", ascending=True).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

diff_csv = Path("Embedding/embedding_diff_summary.csv")
emb_csv = Path("Embedding/embeddings_csv.csv")
emb_img_csv = Path("Embedding/embeddingsImages_csv.csv")

summary_sorted.to_csv(diff_csv, index=False)
pd.DataFrame(E).to_csv(emb_csv, index=False, header=False)
pd.DataFrame(EI).to_csv(emb_img_csv, index=False, header=False)

print(f"✅ Wrote summary to {diff_csv}")
print(f"✅ Wrote embeddings to {emb_csv}")
print(f"✅ Wrote image-augmented embeddings to {emb_img_csv}")
