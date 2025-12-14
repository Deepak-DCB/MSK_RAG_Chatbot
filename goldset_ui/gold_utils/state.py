from pathlib import Path
import pandas as pd
from .io import load_jsonl

# Bucket hints (used for relevant filter)
BUCKET_HINTS = {
    "definition":  ["what is", "overview", "introduction", "summary", "main"],
    "mechanism":   ["mechanism", "pathophysiology", "cause", "why", "how", "compression", "instability"],
    "symptoms":    ["symptom", "sign", "presentation", "manifest", "pain", "numbness", "tingling", "weakness"],
    "diagnosis":   ["diagnos", "assess", "measure", "identify", "examination", "test", "imaging"],
    "treatment":   ["treat", "management", "exercise", "stretch", "protocol", "intervention", "rehab"],
    "red_flags":   ["red flag", "urgent", "danger", "contraindication", "warning"],
}


def resolve_paths(base_dir: Path) -> dict:
    cwd = base_dir.parent  # project root holding data files
    candidates_chunks = [cwd / "MSKArticlesINDEX" / "chunks.parquet", cwd / "chunks.parquet"]
    # 🔧 use your persistent live file as primary gold
    candidates_gold   = [cwd / "goldsetv1.jsonl", cwd / "gold_set.jsonl"]
    candidates_edits  = [cwd / "gold_edits.jsonl"]
    candidates_snap   = [cwd / "gold_set_reviewed.jsonl"]

    def find_first(paths):
        for p in paths:
            if p.exists():
                return p
        return paths[0]

    return {
        "chunks_path": find_first(candidates_chunks),
        "gold_path":   find_first(candidates_gold),
        "edits_path":  find_first(candidates_edits),
        "snapshot_path": find_first(candidates_snap),
    }


def load_chunks(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    text_col = "embed_text" if "embed_text" in df.columns else ("text" if "text" in df.columns else None)
    if not text_col:
        raise ValueError("chunks.parquet missing a text column (embed_text/text)")
    df = df.copy()
    df["section"] = df["section"].fillna("").astype(str)
    df[text_col] = df[text_col].fillna("").astype(str)
    df["source_relpath"] = df["source_relpath"].fillna("").astype(str)
    df["n_words"] = df[text_col].str.split().str.len()
    df.rename(columns={text_col: "text"}, inplace=True)
    return df[["chunk_id", "section", "text", "source_relpath", "n_words", "article_id"]]


def build_idx_map(df: pd.DataFrame):
    return {cid: i for i, cid in enumerate(df["chunk_id"].tolist())}


def load_gold_and_edits(gold_path: Path, edits_path: Path):
    return load_jsonl(gold_path), load_jsonl(edits_path)


def apply_edits_to_gold(gold_list: list, edits_dict: dict | list) -> list:
    if isinstance(edits_dict, list):
        edits_dict = {e.get("id"): e for e in edits_dict}
    id2idx = {g["id"]: i for i, g in enumerate(gold_list)}
    merged = [dict(g) for g in gold_list]
    for qid, ed in edits_dict.items():
        idx = id2idx.get(qid)
        if idx is None:
            continue
        for k, v in ed.items():
            if k == "id":
                continue
            merged[idx][k] = v
    return merged


def build_questions_df(gold_merged: list) -> pd.DataFrame:
    rows = []
    for g in gold_merged:
        rows.append({
            "id": g.get("id"),
            "topic": g.get("topic"),
            "bucket": g.get("bucket"),
            "question": g.get("question"),
            "gt_chunk_id": (g.get("gt_chunk_ids") or [None])[0],
            "source_relpath": g.get("source_relpath"),
        })
    return pd.DataFrame(rows)


def relevant_mask(df: pd.DataFrame, bucket: str):
    hints = BUCKET_HINTS.get(bucket, [])
    if not hints:
        return df["section"].notna()  # all True mask
    low_sec = df["section"].str.lower()
    low_txt = df["text"].str.lower()
    mask = low_sec.str.contains("|".join(map(lambda h: h.replace(" ", "\\s*"), hints)), regex=True)
    mask |= low_txt.str.contains("|".join(map(lambda h: h.replace(" ", "\\s*"), hints)), regex=True)
    return mask