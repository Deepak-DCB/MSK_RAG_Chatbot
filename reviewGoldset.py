#!/usr/bin/env python3
"""
reviewGoldset.py — Streamlined, fast gold-set reviewer (embed_text compatible)

Strict AUTO_TOPIC_BUCKET mode:
• For the 50 canonical questions, topic + bucket are ALWAYS auto-filled
  unless the user has provided a REAL (non-empty, non-manual) override.
• Any blank, "manual", "(manual)", None, or whitespace values are treated as
  missing and replaced by AUTO_TOPIC_BUCKET.
• Eliminates inconsistent UI behavior.

UI Features:
• Full preview of ALL current gold chunks.
• Multi-chunk Save/Add/Remove/Clear.
• 50 rows per page default.
• Full article chunk previews (render full text directly).
• All questions shown as table with row numbers + Go buttons.
• Unique widget keys: no collisions.
• Jump persistence: stable.
• All chunk ID handling uses STRINGs, no `int()` conversion anywhere.
"""

import json
import os
import shutil
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as st


# ========================= Path Helpers =========================

def resolve_paths():
    """
    reviewGoldset.py sits in MSK_Chat/
    MSKArticlesINDEX/ sits in MSK_Chat/MSKArticlesINDEX
    """
    here = Path(__file__).resolve()
    msk_chat = here.parent

    chunks_path    = msk_chat / "MSKArticlesINDEX" / "chunks.parquet"
    gold_path      = msk_chat / "gold_set.jsonl"
    edits_path     = msk_chat / "gold_edits.jsonl"
    snapshot_path  = msk_chat / "gold_set_reviewed.jsonl"

    return chunks_path, gold_path, edits_path, snapshot_path


# ========================= JSONL Helpers =========================

def load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def atomic_write_jsonl(records, out_path: Path, make_backup: bool = True):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if make_backup and out_path.exists():
        ts = time.strftime("%Y%m%d-%H%M%S")
        backup = out_path.with_suffix(out_path.suffix + f".bak.{ts}")
        shutil.copy2(out_path, backup)

    with NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        for rec in records:
            tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_name = tmp.name

    os.replace(tmp_name, out_path)
    return out_path


def load_edits_dict(path: Path):
    if not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "id" in rec:
                out[rec["id"]] = rec
    return out


def append_edit_atomic(edit: dict, edits_path: Path):
    edits_path.parent.mkdir(parents=True, exist_ok=True)
    with edits_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(edit, ensure_ascii=False) + "\n")


def remove_edit_for_id(edits_path: Path, qid: str):
    if not edits_path.exists():
        return
    kept = []
    with edits_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            if rec.get("id") != qid:
                kept.append(rec)
    atomic_write_jsonl(kept, edits_path, make_backup=True)


# ========================= Merge Logic =========================

def apply_edits_to_gold(base_gold, edits_by_id):
    out = []

    for rec in base_gold:
        qid = rec.get("id")
        merged = dict(rec)

        if qid in edits_by_id:
            ed = edits_by_id[qid]
            for k in ["topic", "bucket", "section", "source_relpath",
                      "gt_chunk_ids", "ts"]:
                if k not in ed:
                    continue

                val = ed[k]

                # Strict auto-mapping robustness: ignore blank/manual edits
                if isinstance(val, str):
                    sval = val.strip().lower()
                    if sval in ("", "manual", "(manual)"):
                        continue
                if val is None:
                    continue

                merged[k] = val

        out.append(merged)

    return out


# ========================= AUTO_TOPIC_BUCKET =========================

AUTO_TOPIC_BUCKET = {
    "What is Atlantoaxial Instability (AAI)?": ("AAI", "Definition"),
    "What mechanisms cause Atlantoaxial Instability (AAI)?": ("AAI", "Biomechanics"),
    "What symptoms are characteristic of Atlantoaxial Instability?": ("AAI", "Symptoms"),
    "How is Atlantoaxial Instability clinically assessed?": ("AAI", "Assessment"),
    "What is Craniocervical Instability (CCI)?": ("CCI", "Definition"),
    "What mechanisms cause Craniocervical Instability (CCI)?": ("CCI", "Biomechanics"),
    "How is Craniocervical Instability diagnosed?": ("CCI", "Assessment"),
    "What treatments or exercises are recommended for upper cervical instability?": ("AAI/CCI", "Treatment"),

    "What is Temporomandibular Dysfunction (TMD)?": ("TMD", "Definition"),
    "What mechanisms link TMD with cervical dysfunction?": ("TMD", "Biomechanics"),
    "What symptoms are characteristic of TMD?": ("TMD", "Symptoms"),
    "How is TMD assessed in the MSKNeurology model?": ("TMD", "Assessment"),
    "What mechanisms cause tinnitus related to neck and TMJ dysfunction?": ("Tinnitus", "Biomechanics"),
    "How is cervical-related tinnitus differentiated clinically?": ("Tinnitus", "Assessment"),

    "What is Thoracic Outlet Syndrome (TOS)?": ("TOS", "Definition"),
    "What mechanisms cause neurogenic TOS?": ("TOS", "Biomechanics"),
    "What mechanisms cause arterial TOS?": ("TOS", "Biomechanics"),
    "What mechanisms cause venous TOS?": ("TOS", "Biomechanics"),
    "What symptoms are characteristic of TOS in the MSKNeurology model?": ("TOS", "Symptoms"),
    "How is TOS clinically assessed?": ("TOS", "Assessment"),
    "What are the major compression sites involved in TOS?": ("TOS", "Biomechanics"),
    "What treatments or exercises are recommended for TOS?": ("TOS", "Treatment"),

    "What is Scapular Dyskinesis?": ("Scapular Dyskinesis", "Definition"),
    "What mechanisms cause Scapular Dyskinesis?": ("Scapular Dyskinesis", "Biomechanics"),
    "How does Scapular Dyskinesis affect shoulder stability?": ("Scapular Dyskinesis", "Biomechanics"),
    "What symptoms are characteristic of Scapular Dyskinesis?": ("Scapular Dyskinesis", "Symptoms"),
    "How is Scapular Dyskinesis clinically assessed?": ("Scapular Dyskinesis", "Assessment"),
    "What treatments or exercises are recommended for Scapular Dyskinesis?": ("Scapular Dyskinesis", "Treatment"),

    "What is Vestibular Impairment as described in the MSKNeurology model?": ("Vestibular Impairment", "Definition"),
    "What mechanisms cause Cervicogenic Vestibular Dysfunction?": ("Vestibular Impairment", "Biomechanics"),
    "What symptoms characterize vestibular impairment related to cervical dysfunction?": ("Vestibular Impairment", "Symptoms"),
    "What treatments or exercises are recommended for vestibular impairment?": ("Vestibular Impairment", "Treatment"),

    "What mechanisms contribute to chronic lower back pain in the MSKNeurology model?": ("Chronic Low Back Pain", "Biomechanics"),
    "What is Lumbar Lordosis Mechanics?": ("Lumbar Lordosis", "Definition"),
    "What mechanisms cause abnormal lumbar lordosis?": ("Lumbar Lordosis", "Biomechanics"),
    "What is Lumbar Plexus Compression Syndrome (LPCS)?": ("LPCS", "Definition"),
    "What mechanisms cause Lumbar Plexus Compression Syndrome?": ("LPCS", "Biomechanics"),
    "How is LPCS clinically assessed?": ("LPCS", "Assessment"),

    "What biomechanical mechanisms contribute to chronic hip pain?": ("Hip Pain", "Biomechanics"),
    "What biomechanical factors contribute to knee malalignment?": ("Knee Malalignment", "Biomechanics"),
    "What mechanisms cause hip flexor hypertonicity?": ("Hip Flexor Hypertonicity", "Biomechanics"),
    "What mechanisms cause iliopsoas-related pelvic instability?": ("Iliopsoas Pelvic Instability", "Biomechanics"),

    "What is Chronic Muscle Clenching?": ("Chronic Muscle Clenching", "Definition"),
    "What mechanisms cause Chronic Muscle Clenching?": ("Chronic Muscle Clenching", "Biomechanics"),
    "How is chronic muscle clenching evaluated clinically?": ("Chronic Muscle Clenching", "Assessment"),
    "What treatments or exercises reduce chronic muscle clenching?": ("Chronic Muscle Clenching", "Treatment"),

    "What is Myalgic Encephalomyelitis (ME)?": ("ME", "Definition"),
    "What mechanisms contribute to ME in the MSKNeurology model?": ("ME", "Biomechanics"),
    "What is Postural Orthostatic Tachycardia Syndrome (POTS)?": ("POTS", "Definition"),
    "What mechanisms cause POTS in relation to cervical and autonomic dysfunction?": ("POTS", "Biomechanics"),
}


# ========================= Cached Loaders =========================

@st.cache_data(show_spinner="Loading chunks…")
def load_chunks(path: Path):
    df = pd.read_parquet(path)

    text_col = (
        "embed_text" if "embed_text" in df.columns
        else ("text" if "text" in df.columns else None)
    )
    if not text_col:
        raise ValueError("chunks.parquet missing embed_text/text column")

    df = df.copy()
    df["section"] = df["section"].fillna("").astype(str)
    df["source_relpath"] = df["source_relpath"].fillna("").astype(str)
    df[text_col] = df[text_col].fillna("").astype(str)
    df["n_words"] = df[text_col].str.split().str.len()

    if "chunk_idx" in df.columns:
        df["chunk_idx"] = (
            df["chunk_idx"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .where(lambda s: s.str.fullmatch(r"-?\d+"), "-1")
            .astype(int)
        )
    else:
        df["chunk_idx"] = -1

    df = df.rename(columns={text_col: "text"})
    return df[["chunk_id", "section", "text", "source_relpath", "n_words", "chunk_idx"]]


@st.cache_data(show_spinner="Loading gold set…")
def cached_load_gold(path: Path):
    return load_jsonl(path)


@st.cache_data(show_spinner="Loading edits…")
def cached_load_edits(path: Path, mtime: float):
    return load_edits_dict(path)


@st.cache_data
def build_idx_map(df: pd.DataFrame):
    return {str(cid): i for i, cid in enumerate(df["chunk_id"])}


@st.cache_data
def build_questions_df(gold_merged):
    rows = []
    for g in gold_merged:
        rows.append({
            "id": g.get("id"),
            "topic": (g.get("topic") or "").strip(),
            "bucket": (g.get("bucket") or "").strip(),
            "question": g.get("question"),
            "gt_chunk_id": (g.get("gt_chunk_ids") or [""])[0],
            "gt_count": len(g.get("gt_chunk_ids") or []),
            "source_relpath": g.get("source_relpath"),
        })
    return pd.DataFrame(rows)


# ========================= Helpers =========================

def force_rerun():
    st.rerun()


def _get_current_gt_list(item):
    # Always return list of strings
    return [str(cid) for cid in (item.get("gt_chunk_ids") or [])]


def _order_by_chunk_idx(chunk_ids, chunks_df=None):
    # chunk_ids are strings
    if chunks_df is None:
        return list(chunk_ids)

    sub = chunks_df[chunks_df["chunk_id"].isin(chunk_ids)].copy()
    sub["chunk_idx"] = sub["chunk_idx"].fillna(-1).astype(int)
    sub = sub.sort_values("chunk_idx", kind="mergesort")

    ordered = sub["chunk_id"].tolist()
    missing = [cid for cid in chunk_ids if cid not in ordered]
    return ordered + missing


# ========================= Streamlit App =========================

st.set_page_config(page_title="Gold Set Reviewer", layout="wide")
st.title("🩺 Gold Set Reviewer — Strict Auto-Mapping + String Chunk IDs")

# Load paths
chunks_path, gold_path, edits_path, snapshot_path = resolve_paths()

if not chunks_path.exists() or not gold_path.exists():
    st.error("Missing required files.")
    st.stop()

chunks = load_chunks(chunks_path)
gold = cached_load_gold(gold_path)

# Load edits
edits_mtime = edits_path.stat().st_mtime if edits_path.exists() else 0.0
raw_edits = cached_load_edits(edits_path, edits_mtime)

all_ids = {g["id"] for g in gold}
edits = {k: v for k, v in raw_edits.items() if k in all_ids}

# Merge base + edits
merged = apply_edits_to_gold(gold, edits)


# ========================= Strict AUTO-MAPPING =========================

for rec in merged:
    q = (rec.get("question") or "").strip()
    auto = AUTO_TOPIC_BUCKET.get(q)
    if not auto:
        continue

    auto_topic, auto_bucket = auto

    cur_topic = (rec.get("topic") or "").strip().lower()
    cur_bucket = (rec.get("bucket") or "").strip().lower()

    topic_missing = cur_topic in ("", "manual", "(manual)")
    bucket_missing = cur_bucket in ("", "manual", "(manual)")

    if topic_missing:
        rec["topic"] = auto_topic
    if bucket_missing:
        rec["bucket"] = auto_bucket


idx_map = build_idx_map(chunks)
questions_df = build_questions_df(merged)
merged_ids = [g["id"] for g in merged]
qnum_map = {qid: i+1 for i, qid in enumerate(merged_ids)}

jump_id = st.session_state.pop("_pending_jump_id", None)

# (PART 1 ends here)
# ========================= Sidebar Filters =========================

st.sidebar.header("Filters")

qs = questions_df.copy()

all_topics = sorted([t for t in qs["topic"].unique() if t])
topic_filter = st.sidebar.selectbox("Filter by topic", ["(all)"] + all_topics)

if topic_filter != "(all)":
    qs = qs[qs["topic"] == topic_filter]

all_buckets = sorted([b for b in qs["bucket"].unique() if b])
bucket_filter = st.sidebar.multiselect("Filter by bucket", all_buckets, default=all_buckets)

if bucket_filter:
    qs = qs[qs["bucket"].isin(bucket_filter)]

search_text = st.sidebar.text_input("🔎 Search", "")
if search_text:
    qs = qs[qs["question"].str.contains(search_text, case=False, na=False)]

st.sidebar.write(f"Questions shown: **{len(qs)}**")


# ========================= Questions Table =========================

st.subheader("🗃️ All Questions")

rows_per_page = st.selectbox("Rows per page", [10, 20, 30, 50, 100], index=3)

total = len(qs)
max_page = max(0, (total - 1) // rows_per_page)

if "q_list_page" not in st.session_state:
    st.session_state["q_list_page"] = 0

filters_fp = (topic_filter, tuple(sorted(bucket_filter)), search_text, rows_per_page, total)
if st.session_state.get("_filters_fp") != filters_fp:
    st.session_state["_filters_fp"] = filters_fp
    st.session_state["q_list_page"] = 0
    st.session_state.pop("qtable_state", None)

page = st.session_state["q_list_page"]
page = min(max(page, 0), max_page)

start = page * rows_per_page
end = start + rows_per_page
qs_page = qs.iloc[start:end].reset_index(drop=True)

# Render table
for _, row in qs_page.iterrows():
    qid = row["id"]
    num = qnum_map.get(qid, "?")

    cols = st.columns([0.5, 4, 1.2, 1.2, 0.8, 0.6])
    with cols[0]:
        st.write(f"**{num}**")
    with cols[1]:
        st.write(row["question"])
    with cols[2]:
        st.caption(row["topic"])
    with cols[3]:
        st.caption(row["bucket"])
    with cols[4]:
        st.caption(f"{row['gt_count']} gold(s)")
    with cols[5]:
        if st.button("Go", key=f"go_{qid}_{page}"):
            st.session_state["_pending_jump_id"] = qid
            force_rerun()

# pagination
prev_col, page_col, next_col = st.columns([1, 2, 1])
with prev_col:
    if st.button("⬅️ Prev", disabled=(page <= 0)):
        st.session_state["q_list_page"] = max(page - 1, 0)
        force_rerun()
with page_col:
    st.markdown(f"**Page {page+1} / {max_page+1}**")
with next_col:
    if st.button("Next ➡️", disabled=(page >= max_page)):
        st.session_state["q_list_page"] = min(page + 1, max_page)
        force_rerun()

st.markdown("---")


# ========================= Current Question Section =========================

current_id = jump_id or st.session_state.get("current_qid")
if current_id is None or current_id not in qnum_map:
    if len(qs) == 0:
        st.info("No questions match filters.")
        st.stop()
    current_id = qs.iloc[0]["id"]

st.session_state["current_qid"] = current_id

q_idx = qnum_map[current_id] - 1
item = merged[q_idx]

current_gt_ids = _get_current_gt_list(item)
current_gt_ids_str = set(current_gt_ids)

st.subheader(f"Current Question (#{q_idx+1} of {len(merged)})")
st.markdown(f"**Q:** {item['question']}")

# Topic / bucket editor
col1, col2 = st.columns(2)
with col1:
    new_topic = st.text_input("Topic", value=item.get("topic") or "")
with col2:
    new_bucket = st.text_input("Bucket", value=item.get("bucket") or "")

if st.button("💾 Save Topic + Bucket"):
    edit = {
        "id": item["id"],
        "topic": new_topic,
        "bucket": new_bucket,
        "ts": time.time(),
    }
    append_edit_atomic(edit, edits_path)
    st.session_state["_pending_jump_id"] = item["id"]
    st.success("Updated.")
    force_rerun()

st.caption(f"Topic: {item['topic']} | Bucket: {item['bucket']} | Q {q_idx+1}/{len(merged)}")


# ========================= Gold Chunk Preview =========================

goldsel_key = f"goldsel::{item['id']}"
if goldsel_key not in st.session_state:
    st.session_state[goldsel_key] = set()
gold_selected = st.session_state[goldsel_key]

st.markdown("#### 🌟 Current Gold Chunk Set")

if current_gt_ids:
    for j, cid in enumerate(current_gt_ids, 1):
        cid_str = str(cid)
        row = chunks.iloc[idx_map[cid_str]]

        cols = st.columns([0.9, 7.6])
        with cols[0]:
            sel = st.checkbox(
                f"{j}",
                key=f"chk_gold_{current_id}_{cid_str}",
                value=(cid_str in gold_selected)
            )
            if sel:
                gold_selected.add(cid_str)
            else:
                gold_selected.discard(cid_str)

        with cols[1]:
            st.markdown(
                f"**Chunk {cid_str}** — {row['n_words']} words — {row['source_relpath']} — "
                f"section `{row['section']}`"
            )
            st.info(row["text"])
else:
    st.info("No gold chunks assigned.")

st.markdown("---")


# ========================= Article Chunks =========================

st.markdown("### 📚 Article Chunks — Full Preview")

cur_source = item.get("source_relpath") or ""
all_sources = sorted(chunks["source_relpath"].unique())
default_article = cur_source if cur_source in all_sources else all_sources[0]

article_sel = st.selectbox(
    "Choose article",
    options=all_sources,
    index=all_sources.index(default_article)
)

article_df = chunks[chunks["source_relpath"] == article_sel].copy()
article_df = article_df.sort_values("chunk_idx", kind="mergesort")

multi_key = f"multi::{item['id']}::{article_sel}"
if multi_key not in st.session_state:
    st.session_state[multi_key] = set(
        cid for cid in current_gt_ids if chunks.iloc[idx_map[cid]]["source_relpath"] == article_sel
    )
article_selected = st.session_state[multi_key]

st.info(f"Showing all {len(article_df)} chunks.")


def _toolbar(label, multi_key, goldsel_key):
    article_selected_local = st.session_state[multi_key]
    gold_selected_local = st.session_state[goldsel_key]
    current_gt_local = _get_current_gt_list(item)

    base_key = f"{label}_{item['id']}_{article_sel}"

    m1, m2, _, m4, m5 = st.columns([2.3, 2.1, 0.3, 3.2, 1.6])

    # SAVE
    with m1:
        if st.button(f"💾 Save ({label})", key=f"save_{base_key}"):
            keep_other = [
                cid for cid in current_gt_local
                if chunks.iloc[idx_map[cid]]["source_relpath"] != article_sel
            ]
            ordered_local = _order_by_chunk_idx(list(article_selected_local), article_df)
            new_ids = ordered_local + keep_other

            if ordered_local:
                first = ordered_local[0]
                r0 = chunks.iloc[idx_map[first]]
                sec_val = r0["section"]
                src_val = r0["source_relpath"]
            else:
                sec_val = item.get("section", "")
                src_val = item.get("source_relpath", article_sel)

            edit = {
                "id": item["id"],
                "gt_chunk_ids": new_ids,
                "section": sec_val,
                "source_relpath": src_val,
                "ts": time.time(),
            }
            append_edit_atomic(edit, edits_path)
            st.session_state["_pending_jump_id"] = item["id"]
            st.session_state[goldsel_key] = set()
            st.success("Saved.")
            force_rerun()

    # ADD
    with m2:
        if st.button(f"➕ Add ({label})", key=f"add_{base_key}"):
            if not article_selected_local:
                st.warning("Nothing selected.")
            else:
                existing = set(current_gt_local)
                for cid in article_selected_local:
                    existing.add(cid)
                ordered = _order_by_chunk_idx(list(existing), chunks_df=chunks)

                sec_val = next(
                    (r["section"] for _, r in article_df.iterrows()
                     if str(r["chunk_id"]) in article_selected_local),
                    ""
                )

                edit = {
                    "id": item["id"],
                    "gt_chunk_ids": ordered,
                    "source_relpath": article_sel,
                    "section": sec_val,
                    "ts": time.time(),
                }
                append_edit_atomic(edit, edits_path)
                st.session_state["_pending_jump_id"] = item["id"]
                st.session_state[goldsel_key] = set()
                st.success("Added.")
                force_rerun()

    # REMOVE
    with m4:
        if st.button(f"➖ Remove ({label})", key=f"remove_{base_key}"):
            if not gold_selected_local:
                st.warning("No gold selected.")
            else:
                remaining = [
                    cid for cid in current_gt_local if cid not in gold_selected_local
                ]

                st.session_state[multi_key] = {
                    cid for cid in article_selected_local if cid not in gold_selected_local
                }

                edit = {
                    "id": item["id"],
                    "gt_chunk_ids": _order_by_chunk_idx(remaining),
                    "source_relpath": item.get("source_relpath", article_sel),
                    "section": item.get("section", ""),
                    "ts": time.time(),
                }
                append_edit_atomic(edit, edits_path)
                st.session_state["_pending_jump_id"] = item["id"]
                st.session_state[goldsel_key] = set()
                st.success("Removed.")
                force_rerun()

    # CLEAR
    with m5:
        if st.button(f"🧹 Clear ({label})", key=f"clear_{base_key}"):
            st.session_state[multi_key] = set()
            st.session_state[goldsel_key] = set()
            st.success("Cleared.")


# Top Toolbar
_toolbar("TOP", multi_key, goldsel_key)

# Render chunks
current_gt_ids_str = set(current_gt_ids)

for i, r in article_df.reset_index(drop=True).iterrows():
    cid = str(r["chunk_id"])
    is_sel = cid in article_selected
    is_gold = cid in current_gt_ids_str

    st.markdown(f"### {'⭐ ' if is_gold else ''}Chunk {i+1}/{len(article_df)}")

    cols = st.columns([0.8, 7.2])
    with cols[0]:
        chk = st.checkbox(
            "Select",
            value=is_sel,
            key=f"chk_{item['id']}_{article_sel}_{cid}"
        )
        if chk:
            article_selected.add(cid)
        else:
            article_selected.discard(cid)

    with cols[1]:
        st.markdown(
            f"**Chunk ID:** {cid} | idx={r['chunk_idx']} | words={r['n_words']} | "
            f"section `{r['section']}`"
        )
        st.markdown(f"**Source:** {r['source_relpath']}")
        st.markdown("**Text:**")
        st.info(r["text"])

    st.markdown("---")

# Bottom Toolbar
_toolbar("BOTTOM", multi_key, goldsel_key)


# ========================= Revert =========================

if st.button("↩️ Revert this question to original"):
    remove_edit_for_id(edits_path, item["id"])
    st.success("Reverted.")
    st.session_state["_pending_jump_id"] = item["id"]
    force_rerun()


# ========================= Snapshot =========================

if st.button("💾 Save Snapshot"):
    atomic_write_jsonl(merged, snapshot_path, make_backup=True)
    st.success(f"Snapshot saved → {snapshot_path}")
