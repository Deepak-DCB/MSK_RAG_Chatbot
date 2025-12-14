#!/usr/bin/env python3
"""
merge_gold_for_eval.py — Merge base gold_set.jsonl + gold_edits.jsonl

Produces:
    gold_set_merged_for_eval.jsonl

This creates the *actual* latest reviewed gold set, identical to what
reviewGoldset.py shows after applying all edits and auto-topic/bucket mapping.

Run:
    python merge_gold_for_eval.py
"""

import json
from pathlib import Path


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

HERE = Path(__file__).resolve().parent.parent   # MSK_Chat/
EVAL = Path(__file__).resolve().parent          # MSK_Chat/Eval/

BASE_GOLD   = HERE / "gold_set.jsonl"
EDITS_FILE  = HERE / "gold_edits.jsonl"
OUT_FILE    = EVAL / "gold_set_merged_for_eval.jsonl"


# ------------------------------------------------------------
# AUTO_TOPIC_BUCKET (identical to reviewGoldset.py)
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# JSONL loader
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Auto topic/bucket fill
# ------------------------------------------------------------

def auto_apply_topic_bucket(rec):
    """
    Ensure topic + bucket fields are filled using AUTO_TOPIC_BUCKET.
    This matches reviewGoldset.py behavior exactly.
    """
    q = (rec.get("question") or "").strip()
    if q not in AUTO_TOPIC_BUCKET:
        return rec

    auto_topic, auto_bucket = AUTO_TOPIC_BUCKET[q]

    cur_topic = (rec.get("topic") or "").strip().lower()
    cur_bucket = (rec.get("bucket") or "").strip().lower()

    topic_missing  = cur_topic in ("", "manual", "(manual)")
    bucket_missing = cur_bucket in ("", "manual", "(manual)")

    if topic_missing:
        rec["topic"] = auto_topic
    if bucket_missing:
        rec["bucket"] = auto_bucket

    return rec


# ------------------------------------------------------------
# Merge edits + auto topics
# ------------------------------------------------------------

def apply_edits_to_gold(base_gold, edits_by_id):
    out = []

    for rec in base_gold:
        qid = rec.get("id")
        merged = dict(rec)

        # Apply edits
        if qid in edits_by_id:
            ed = edits_by_id[qid]
            for k in ["topic", "bucket", "section", "source_relpath",
                      "gt_chunk_ids", "ts"]:
                if k not in ed:
                    continue

                val = ed[k]

                # reviewer rule: ignore blank/manual/edit-null
                if isinstance(val, str):
                    sval = val.strip().lower()
                    if sval in ("", "manual", "(manual)"):
                        continue
                if val is None:
                    continue

                merged[k] = val

        # Apply auto topic/bucket
        merged = auto_apply_topic_bucket(merged)

        out.append(merged)

    return out


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    if not BASE_GOLD.exists():
        raise SystemExit(f"❌ Missing base gold: {BASE_GOLD}")

    base = load_jsonl(BASE_GOLD)
    edits = load_jsonl(EDITS_FILE)
    edits_by_id = {e["id"]: e for e in edits if "id" in e}

    print(f"Loaded {len(base)} base records")
    print(f"Loaded {len(edits_by_id)} edit records")

    merged = apply_edits_to_gold(base, edits_by_id)

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Wrote merged gold set → {OUT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
