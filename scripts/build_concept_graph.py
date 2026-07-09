#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "VectorDB"))

from graph_paths import build_mechanism_paths  # noqa: E402
from graph_vocab import (  # noqa: E402
    CLAIM_STRENGTHS,
    CLAIM_TYPES,
    CLINICAL_RISKS,
    RELATION_TYPES,
    SCHEMA_VERSION,
    SUPPORT_LEVELS,
    all_entities,
    canonical_node_id,
    detect_entities,
)

HIER_DIR = PROJECT_ROOT / "MSKArticlesINDEX" / "hierarchical"
GRAPH_DIR = PROJECT_ROOT / "MSKArticlesINDEX" / "graph"

SYMPTOMS = {
    "numbness", "paresthesia", "tingling", "neuralgia", "ulnar neuralgia", "neck pain", "shoulder pain", "pain between shoulder blades", "dorsal scapular pain", "headache",
    "dizziness", "dyspnea", "globus", "TOS-like symptoms", "tinnitus", "ear pressure", "hearing symptoms",
    "vestibular symptoms", "fatigue", "orthostatic symptoms", "autonomic symptoms", "hip pain", "groin pain", "pelvic pain", "knee pain",
    "low back pain", "jaw pain", "brain fog", "chest pain", "tremor", "Raynaud's syndrome", "causalgia",
}
NERVES = {
    "brachial plexus", "cervical plexus", "lumbar plexus", "lumbosacral plexus", "sacral plexus", "pudendal nerve",
    "inferior hypogastric plexus", "femoral nerve", "sciatic nerve", "ilioinguinal nerve", "iliohypogastric nerve",
    "genitofemoral nerve", "lateral femoral cutaneous nerve", "ulnar nerve", "median nerve", "radial nerve", "axillary nerve",
    "dorsal scapular nerve", "phrenic nerve", "nerve root", "vestibulocochlear nerve", "trigeminal nerve", "facial nerve",
    "glossopharyngeal nerve", "vagus nerve", "auriculotemporal nerve",
    "accessory nerve", "occipital nerves", "mandibular nerve", "maxillary nerve", "ophthalmic nerve", "buccal nerve", "lingual nerve",
}
SPACES = {
    "costoclavicular space", "costoclavicular passage", "interscalene triangle", "thoracic outlet", "subcoracoid space",
    "retropectoralis minor space", "foramen magnum", "Alcock's canal", "pelvic floor", "subacromial space",
    "middle ear", "eustachian tube", "tympanic plexus", "vestibular system", "cochlea", "semicircular canals", "utricle", "sacculus", "jugular foramen",
}
VESSELS = {"subclavian artery", "subclavian vein", "vertebral artery", "vertebral vein", "internal jugular vein", "external jugular vein", "internal carotid artery", "basilar artery", "venous sinus"}
ANATOMICAL_SUPPLY_TARGETS = {
    "atlas", "axis", "clavicle", "first rib", "scapula", "humerus", "acromion", "coracoid process", "femur", "acetabulum",
    "pelvis", "lumbar spine", "sacrum", "tibia", "patella", "fibula", "mandible", "maxilla", "temporal bone", "odontoid process", "brainstem", "transverse ligament", "alar ligament", "tectorial membrane", *SPACES,
}
INNERVATION_TARGETS = ANATOMICAL_SUPPLY_TARGETS | {
    "scalene", "anterior scalene", "middle scalene", "levator scapulae", "trapezius", "pectoralis minor", "sternocleidomastoid",
    "serratus anterior", "rhomboids", "teres minor", "teres major", "rotator cuff", "supraspinatus", "subscapularis",
    "psoas", "iliacus", "gluteus maximus", "gluteus medius", "piriformis", "obturator internus", "tensor fascia latae",
    "quadratus lumborum", "lumbar extensors", "hamstrings", "quadriceps", "gastrocnemius", "popliteus", "pterygoids",
    "masseter", "temporalis", "suboccipitals", "longus colli", "longus capitis",
    "lateral pterygoid", "medial pterygoid", "tensor tympani", "tensor veli palatini",
}
UPPER_COMPRESSION_SOURCES = {"scalene", "pectoralis minor", "clavicle", "first rib", "costoclavicular narrowing", "costoclavicular space", "costoclavicular passage", "interscalene triangle", "thoracic outlet", "retropectoralis minor space"}
UPPER_COMPRESSION_TARGETS = {"brachial plexus", "subclavian artery", "subclavian vein", "cervical plexus", "dorsal scapular nerve", "ulnar nerve", "median nerve", "radial nerve", "axillary nerve", "musculocutaneous nerve", "phrenic nerve", "brachial plexus compression", "neurovascular compression"}
LOWER_COMPRESSION_SOURCES = {"psoas", "piriformis", "obturator internus", "pelvic floor", "Alcock's canal", "lumbar spine"}
LOWER_COMPRESSION_TARGETS = {"lumbar plexus", "lumbosacral plexus", "sacral plexus", "pudendal nerve", "inferior hypogastric plexus", "femoral nerve", "sciatic nerve", "ilioinguinal nerve", "iliohypogastric nerve", "genitofemoral nerve", "lateral femoral cutaneous nerve", "nerve root"}
CRANIAL_COMPRESSION_SOURCES = {"atlas", "axis", "mandible", "foramen magnum", "jugular foramen", "jugular outlet obstruction", "temporomandibular joint", "lateral pterygoid", "suboccipitals", "trapezius"}
CRANIAL_COMPRESSION_TARGETS = {"vertebral artery", "vertebral vein", "internal jugular vein", "internal carotid artery", "basilar artery", "brainstem compression", "venous outflow impairment", "vagus nerve", "accessory nerve", "glossopharyngeal nerve", "auriculotemporal nerve", "trigeminal nerve", "mandibular nerve", "buccal nerve", "lingual nerve", "occipital nerves"}
APPROVED_NERVE_MUSCLE_RELATIONS = {
    ("accessory nerve", "sternocleidomastoid"),
    ("accessory nerve", "trapezius"),
    ("trigeminal nerve", "tensor tympani"),
    ("trigeminal nerve", "tensor veli palatini"),
    ("trigeminal nerve", "lateral pterygoid"),
    ("trigeminal nerve", "medial pterygoid"),
    ("mandibular nerve", "lateral pterygoid"),
    ("mandibular nerve", "medial pterygoid"),
    ("buccal nerve", "lateral pterygoid"),
    ("lingual nerve", "lateral pterygoid"),
    ("vagus nerve", "tensor veli palatini"),
}
DISALLOWED_SAME_SPAN_MENTIONS = {
    frozenset(("auriculotemporal nerve", "brachial plexus")),
    frozenset(("phrenic nerve", "auriculotemporal nerve")),
    frozenset(("vagus nerve", "lumbar plexus")),
    frozenset(("vagus nerve", "lumbar plexus compression syndrome")),
    frozenset(("lumbar plexus", "brachial plexus")),
    frozenset(("lumbar plexus compression syndrome", "brachial plexus")),
}
APPROVED_STABILIZES = {
    ("rotator cuff", "glenohumeral joint"),
    ("supraspinatus", "glenohumeral joint"),
    ("subscapularis", "glenohumeral joint"),
    ("trapezius", "scapula"),
    ("serratus anterior", "scapula"),
    ("suboccipitals", "atlas"),
    ("longus colli", "atlas"),
    ("longus capitis", "atlas"),
    ("longus colli", "axis"),
    ("longus capitis", "axis"),
    ("suboccipitals", "atlantoaxial joint"),
    ("longus colli", "atlantoaxial joint"),
    ("longus capitis", "atlanto-occipital joint"),
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_id(prefix: str, parts: Iterable[Any]) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def sentence_window(text: str, names: Iterable[str]) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", normalize_text(text))
    wanted = [n.lower() for n in names]
    hits = [s for s in sentences if any(name in s.lower() for name in wanted)]
    out = " ".join(hits[:2]) or normalize_text(text)
    return out[:700]


def add_edge(
    edge_map: Dict[Tuple[str, str, str], Dict[str, Any]],
    source: str,
    target: str,
    relation_type: str,
    support_level: str,
    claim_strength: str,
    clinical_risk: str,
    span: Dict[str, Any],
    evidence_names: Iterable[str],
    notes: str = "",
) -> None:
    if source == target:
        return
    assert relation_type in RELATION_TYPES
    assert support_level in SUPPORT_LEVELS
    assert claim_strength in CLAIM_STRENGTHS
    assert clinical_risk in CLINICAL_RISKS
    source_id = canonical_node_id(source)
    target_id = canonical_node_id(target)
    key = (source_id, target_id, relation_type)
    existing = edge_map.get(key)
    if existing is None:
        existing = {
            "edge_id": stable_id("edge", key),
            "source_node_id": source_id,
            "target_node_id": target_id,
            "relation_type": relation_type,
            "support_level": support_level,
            "claim_strength": claim_strength,
            "clinical_risk": clinical_risk,
            "source_span_ids": [],
            "source_section_ids": [],
            "source_article_ids": [],
            "evidence_text": "",
            "notes": notes,
            "schema_version": SCHEMA_VERSION,
        }
        edge_map[key] = existing
    span_id = span.get("span_id")
    if span_id and span_id not in existing["source_span_ids"]:
        existing["source_span_ids"].append(span_id)
    section_id = span.get("section_id")
    if section_id and section_id not in existing["source_section_ids"]:
        existing["source_section_ids"].append(section_id)
    article_id = span.get("article_id")
    if article_id and article_id not in existing["source_article_ids"]:
        existing["source_article_ids"].append(article_id)
    if not existing["evidence_text"]:
        existing["evidence_text"] = sentence_window(span.get("text", ""), evidence_names)


def certainty_for(text: str) -> Tuple[str, str]:
    low = text.lower()
    if re.search(r"\b(may|might|could|can|possible|potential|often|usually|suggest|associated)\b", low):
        return "indirect", "moderate"
    return "direct", "moderate"


def relation_edges_for_span(edge_map: Dict[Tuple[str, str, str], Dict[str, Any]], span: Dict[str, Any], entities: Set[str]) -> None:
    text = normalize_text(span.get("text", ""))
    low = text.lower()
    names = sorted(entities)

    for source in names:
        for target in names:
            if source >= target:
                continue
            if frozenset((source, target)) in DISALLOWED_SAME_SPAN_MENTIONS:
                continue
            add_edge(edge_map, source, target, "mentioned_with", "weak", "weak", "low", span, [source, target], "same evidence span co-mention")

    def connect_sources_to_targets(sources: Iterable[str], targets: Iterable[str], rel: str, support: str, strength: str, risk: str = "low") -> None:
        for s in sources:
            for t in targets:
                if s != t and frozenset((s, t)) not in DISALLOWED_SAME_SPAN_MENTIONS:
                    add_edge(edge_map, s, t, rel, support, strength, risk, span, [s, t])

    def connect_approved_pairs(pairs: Iterable[Tuple[str, str]], rel: str, support: str, strength: str, risk: str = "low") -> None:
        for s, t in pairs:
            if s in names and t in names and s != t and frozenset((s, t)) not in DISALLOWED_SAME_SPAN_MENTIONS:
                add_edge(edge_map, s, t, rel, support, strength, risk, span, [s, t])

    if any(term in low for term in ["compress", "compression", "compressed", "entrap"]):
        support, strength = certainty_for(low)
        rel = "may_compress" if support != "direct" or "may" in low or "can" in low else "compresses"
        pairs = []
        pairs.extend((s, t) for s in names if s in UPPER_COMPRESSION_SOURCES for t in names if t in UPPER_COMPRESSION_TARGETS)
        pairs.extend((s, t) for s in names if s in LOWER_COMPRESSION_SOURCES for t in names if t in LOWER_COMPRESSION_TARGETS)
        pairs.extend((s, t) for s in names if s in CRANIAL_COMPRESSION_SOURCES for t in names if t in CRANIAL_COMPRESSION_TARGETS)
        connect_approved_pairs(pairs, rel, support, strength, "medium")

    if any(term in low for term in ["narrow", "narrows", "narrowing", "reduced space", "less space"]):
        sources = [e for e in names if e in {"scapular depression", "clavicular depression", "clavicle", "first rib", "scalene dysfunction", "anterior scapular tilt", "posterior pelvic tilt", "mandibular retraction"}]
        targets = [e for e in names if e in SPACES or e == "costoclavicular narrowing"]
        connect_sources_to_targets(sources, targets, "narrows", "direct", "moderate", "medium")

    if any(term in low for term in ["passes through", "travel through", "runs through", "traverse"]):
        connect_sources_to_targets([e for e in names if e in NERVES or e in VESSELS], [e for e in names if e in SPACES], "passes_through", "direct", "strong")

    if "innervat" in low:
        connect_approved_pairs(APPROVED_NERVE_MUSCLE_RELATIONS, "innervates", "direct", "strong")

    if any(term in low for term in ["controls", "control the", "controlled by"]):
        connect_approved_pairs(APPROVED_NERVE_MUSCLE_RELATIONS, "innervates", "direct", "strong")

    if "suppl" in low:
        connect_sources_to_targets([e for e in names if e in VESSELS], [e for e in names if e in ANATOMICAL_SUPPLY_TARGETS], "supplies", "direct", "strong")

    if any(term in low for term in ["may cause", "can cause", "cause", "causes", "produce", "symptoms", "results in", "lead to", "leads to"]):
        support, strength = certainty_for(low)
        rel = "may_produce" if any(s in entities for s in SYMPTOMS) else "may_contribute_to"
        sources = [e for e in names if e not in SYMPTOMS]
        targets = [e for e in names if e in SYMPTOMS]
        connect_sources_to_targets(sources, targets, rel, support if support != "direct" else "indirect", strength, "medium" if targets else "low")

    if "dorsal scapular nerve" in entities:
        span_id = str(span.get("span_id") or "")
        if span_id == "67c0c170eb3517fcf9c82dd275f4e709" and "pain between" in low and "shoulder blades" in low:
            add_edge(edge_map, "dorsal scapular nerve", "pain between shoulder blades", "may_produce", "indirect", "moderate", "medium", span, ["dorsal scapular nerve", "pain between shoulder blades"])
        if span_id == "bce1016b5782bdd120224975c2159999" and "periscapular" in low:
            add_edge(edge_map, "dorsal scapular nerve", "dorsal scapular pain", "may_produce", "indirect", "moderate", "medium", span, ["dorsal scapular nerve", "dorsal scapular pain"])

    if any(term in low for term in ["contribute", "contributes", "predispose", "influence", "affect"]):
        sources = [e for e in names if e not in SYMPTOMS]
        targets = [e for e in names if e in SPACES or e in NERVES or e in SYMPTOMS or e in {"thoracic outlet mechanics", "scalene dysfunction", "first rib mechanics"}]
        connect_sources_to_targets(sources, targets, "may_contribute_to", "indirect", "moderate", "medium")

    if any(term in low for term in ["depress", "depression", "descent"]):
        sources = [e for e in names if e in {"scapular depression", "clavicular depression", "trapezius", "pectoralis minor"}]
        targets = [e for e in names if e in {"scapula", "clavicle", "first rib", "costoclavicular space"}]
        connect_sources_to_targets(sources, targets, "depresses", "direct", "moderate", "low")

    if any(term in low for term in ["elevat", "raise"]):
        connect_sources_to_targets([e for e in names if e in {"scalene", "anterior scalene", "middle scalene"}], [e for e in names if e == "first rib"], "elevates", "direct", "moderate")

    if "reproduce" in low and "symptom" in low:
        connect_sources_to_targets([e for e in names if e in {"provocative testing"}], [e for e in names if e in SYMPTOMS or e in NERVES], "tests_for", "direct", "moderate")

    if any(term in low for term in ["test", "tests", "assess", "diagnos", "image", "mri", "ct", "block", "tinel"]):
        connect_sources_to_targets([e for e in names if e in {"provocative testing", "manual muscle testing", "FABER test", "dynamic imaging", "nerve block", "Tinel's sign"}], [e for e in names if e in SYMPTOMS or e in NERVES or e in {"atlantoaxial instability", "craniocervical instability", "intracranial hypertension", "pudendal neuralgia", "femoroacetabular impingement"}], "tests_for", "direct", "moderate")

    if any(term in low for term in ["strengthen", "strengthening", "exercise", "rehab", "rehabilitation", "posture", "correction"]):
        connect_sources_to_targets([e for e in names if e in {"strengthening", "postural rehabilitation", "awareness training"}], [e for e in names if e in NERVES or e in SYMPTOMS or e in INNERVATION_TARGETS], "may_contribute_to", "indirect", "moderate")

    if any(term in low for term in ["stabiliz", "stability", "support"]):
        connect_approved_pairs(APPROVED_STABILIZES, "stabilizes", "direct", "moderate")


def add_bridge_edges(edge_map: Dict[Tuple[str, str, str], Dict[str, Any]], spans: List[Dict[str, Any]], detected_by_span: Dict[str, Set[str]]) -> None:
    by_section: Dict[str, List[Dict[str, Any]]] = {}
    for span in spans:
        by_section.setdefault(str(span.get("section_id")), []).append(span)
    bridge_pairs = [
        ("scapular depression", "clavicular depression", "may_contribute_to"),
        ("clavicular depression", "costoclavicular narrowing", "may_contribute_to"),
        ("costoclavicular narrowing", "costoclavicular space", "associated_with"),
        ("costoclavicular space", "brachial plexus", "may_compress"),
        ("costoclavicular space", "subclavian artery", "may_compress"),
        ("costoclavicular space", "subclavian vein", "may_compress"),
        ("brachial plexus", "paresthesia", "may_produce"),
        ("brachial plexus", "neuralgia", "may_produce"),
        ("brachial plexus", "numbness", "may_produce"),
        ("scalene dysfunction", "first rib mechanics", "may_contribute_to"),
        ("first rib mechanics", "interscalene triangle", "may_contribute_to"),
        ("interscalene triangle", "brachial plexus", "may_compress"),
        ("forward head posture", "altered breathing mechanics", "may_contribute_to"),
        ("swayback posture", "altered breathing mechanics", "may_contribute_to"),
        ("slouching", "altered breathing mechanics", "may_contribute_to"),
        ("altered breathing mechanics", "scalene dysfunction", "may_contribute_to"),
        ("scalene dysfunction", "thoracic outlet mechanics", "may_contribute_to"),
        ("levator scapulae", "cervical plexus entrapment", "may_contribute_to"),
        ("middle scalene", "cervical plexus entrapment", "may_contribute_to"),
        ("cervical plexus entrapment", "neck pain", "may_produce"),
        ("cervical plexus entrapment", "headache", "may_produce"),
        ("costoclavicular space", "dorsal scapular nerve", "may_compress"),
        ("thoracic outlet syndrome", "dorsal scapular nerve", "may_contribute_to"),
        ("dorsal scapular nerve", "pain between shoulder blades", "may_produce"),
        ("dorsal scapular nerve", "neck pain", "may_produce"),
        ("dorsal scapular nerve", "neuralgia", "may_produce"),
        ("brachial plexus irritation", "sensitized nervous chain", "may_contribute_to"),
        ("sensitized nervous chain", "distal nerve vulnerability", "may_contribute_to"),
        ("distal nerve vulnerability", "ulnar nerve", "may_contribute_to"),
        ("distal nerve vulnerability", "median nerve", "may_contribute_to"),
        ("distal nerve vulnerability", "radial nerve", "may_contribute_to"),
        ("ulnar nerve", "ulnar neuralgia", "may_produce"),
        ("posterior pelvic tilt", "lumbar lordosis", "may_contribute_to"),
        ("posterior pelvic tilt", "disc loading", "may_contribute_to"),
        ("lumbar lordosis", "lumbar spine", "associated_with"),
        ("disc loading", "disc herniation", "may_contribute_to"),
        ("disc herniation", "nerve root", "may_compress"),
        ("disc herniation", "low back pain", "may_produce"),
        ("nerve root", "low back pain", "may_produce"),
        ("psoas", "lumbar plexus", "may_compress"),
        ("lumbar spine", "lumbar plexus", "associated_with"),
        ("lumbar plexus", "femoral nerve", "may_contribute_to"),
        ("lumbar plexus", "genitofemoral nerve", "may_contribute_to"),
        ("lumbar plexus", "ilioinguinal nerve", "may_contribute_to"),
        ("lumbar plexus", "lateral femoral cutaneous nerve", "may_contribute_to"),
        ("femoral nerve", "groin pain", "may_produce"),
        ("genitofemoral nerve", "groin pain", "may_produce"),
        ("ilioinguinal nerve", "groin pain", "may_produce"),
        ("psoas", "anterior femoral glide", "may_contribute_to"),
        ("posterior pelvic tilt", "anterior femoral glide", "may_contribute_to"),
        ("anterior femoral glide", "femoroacetabular impingement", "may_contribute_to"),
        ("femoroacetabular impingement", "labral impingement", "may_contribute_to"),
        ("femoroacetabular impingement", "hip pain", "may_produce"),
        ("labral impingement", "groin pain", "may_produce"),
        ("posterior pelvic tilt", "functional varus", "may_contribute_to"),
        ("functional varus", "posterior tibial glide", "may_contribute_to"),
        ("functional valgus", "tibial external rotation", "may_contribute_to"),
        ("tibial external rotation", "patella", "may_contribute_to"),
        ("posterior tibial glide", "patella", "may_contribute_to"),
        ("anterior tibial glide", "jumper's knee", "may_contribute_to"),
        ("jumper's knee", "knee pain", "may_produce"),
        ("patella", "patellofemoral pain syndrome", "may_contribute_to"),
        ("patellofemoral pain syndrome", "knee pain", "may_produce"),
        ("pelvic floor", "pudendal nerve", "may_compress"),
        ("Alcock's canal", "pudendal nerve", "may_compress"),
        ("piriformis", "pudendal nerve", "may_compress"),
        ("obturator internus", "pudendal nerve", "may_compress"),
        ("pudendal nerve", "pudendal neuralgia", "may_contribute_to"),
        ("pudendal neuralgia", "pelvic pain", "may_produce"),
        ("inferior hypogastric plexus", "inferior hypogastric plexopathy", "may_contribute_to"),
        ("inferior hypogastric plexopathy", "pelvic pain", "may_produce"),
        ("scapular dyskinesis", "subacromial space", "may_contribute_to"),
        ("scapular depression", "scapular dyskinesis", "associated_with"),
        ("scapular winging", "scapular dyskinesis", "associated_with"),
        ("subacromial space", "shoulder impingement", "may_contribute_to"),
        ("shoulder impingement", "rotator cuff", "may_contribute_to"),
        ("rotator cuff", "shoulder pain", "may_produce"),
        ("scapular dyskinesis", "motor control", "associated_with"),
        ("motor control", "strengthening", "requires_caution"),
        ("strengthening", "shoulder pain", "may_contribute_to"),
        ("protective bracing", "manual muscle testing", "may_contribute_to"),
        ("protective bracing", "altered breathing mechanics", "may_contribute_to"),
        ("breath holding", "altered breathing mechanics", "associated_with"),
        ("altered breathing mechanics", "fatigue", "may_contribute_to"),
        ("protective bracing", "fatigue", "may_contribute_to"),
        ("atlas", "vertebral artery", "may_contribute_to"),
        ("axis", "vertebral artery", "may_contribute_to"),
        ("atlas instability", "vertebral artery", "may_contribute_to"),
        ("vertebral artery", "dizziness", "may_produce"),
        ("atlantoaxial instability", "brainstem compression", "may_contribute_to", "high"),
        ("craniocervical instability", "brainstem compression", "may_contribute_to", "high"),
        ("brainstem compression", "brainstem compromise", "red_flag_for", "high"),
        ("brainstem compromise", "progressive neurologic deficit", "red_flag_for", "high"),
        ("vertebrobasilar insufficiency", "dizziness", "red_flag_for", "high"),
        ("vertebrobasilar insufficiency", "progressive neurologic deficit", "red_flag_for", "high"),
        ("internal jugular vein", "venous outflow impairment", "may_contribute_to"),
        ("jugular outlet obstruction", "venous outflow impairment", "may_contribute_to"),
        ("venous sinus stenosis", "intracranial hypertension", "may_contribute_to"),
        ("venous outflow impairment", "intracranial hypertension", "may_contribute_to"),
        ("CSF leak", "CSF pressure", "may_contribute_to"),
        ("CSF pressure", "headache", "may_produce"),
        ("CSF pressure", "dizziness", "may_produce"),
        ("CSF pressure", "vestibular symptoms", "may_produce"),
        ("intracranial hypertension", "papilledema", "red_flag_for", "high"),
        ("intracranial hypertension", "tinnitus", "may_produce"),
        ("intracranial hypertension", "vestibular symptoms", "may_produce"),
        ("venous outflow impairment", "migraine", "may_contribute_to"),
        ("craniovascular hyperperfusion", "POTS", "may_contribute_to"),
        ("POTS", "orthostatic symptoms", "may_produce"),
        ("POTS", "autonomic symptoms", "may_produce"),
        ("vascular TOS", "sympathetic involvement", "may_contribute_to"),
        ("subclavian vein", "vascular TOS", "associated_with"),
        ("sympathetic involvement", "autonomic symptoms", "may_produce"),
        ("ME/CFS", "fatigue", "associated_with"),
        ("ME/CFS", "brain fog", "associated_with"),
        ("temporomandibular joint", "eustachian tube", "may_contribute_to"),
        ("temporomandibular joint", "tympanic plexus", "may_contribute_to"),
        ("temporomandibular joint", "auriculotemporal nerve", "may_compress"),
        ("temporomandibular joint", "trigeminal nerve", "may_contribute_to"),
        ("mouth breathing", "temporomandibular joint", "may_contribute_to"),
        ("trigeminal nerve", "mandibular nerve", "may_contribute_to"),
        ("trigeminal nerve", "auriculotemporal nerve", "may_contribute_to"),
        ("trigeminal nerve", "tensor tympani", "innervates"),
        ("trigeminal nerve", "tensor veli palatini", "innervates"),
        ("trigeminal nerve", "tinnitus", "may_produce"),
        ("trigeminal nerve", "ear pressure", "may_contribute_to"),
        ("mandibular nerve", "eustachian tube", "may_contribute_to"),
        ("auriculotemporal nerve", "ear pressure", "may_contribute_to"),
        ("mouth breathing", "mandibular retraction", "may_contribute_to"),
        ("tongue posture", "mandibular retraction", "may_contribute_to"),
        ("occlusion", "mandibular retraction", "may_contribute_to"),
        ("maxilla", "mandibular retraction", "may_contribute_to"),
        ("mandibular retraction", "temporomandibular joint", "may_contribute_to"),
        ("temporomandibular joint", "disc displacement", "may_contribute_to"),
        ("disc displacement", "jaw pain", "may_produce"),
        ("tympanic plexus", "tinnitus", "may_contribute_to"),
        ("eustachian tube", "ear pressure", "may_produce"),
        ("eustachian tube", "tinnitus", "may_contribute_to"),
        ("suboccipitals", "occipital nerves", "may_compress"),
        ("trapezius", "occipital nerves", "may_compress"),
        ("occipital nerves", "neuralgia", "may_produce"),
        ("occipital nerves", "headache", "may_produce"),
        ("occipital nerves", "neck pain", "may_produce"),
        ("vestibular system", "vestibular symptoms", "may_produce"),
        ("vestibulocochlear nerve", "vestibular system", "associated_with"),
        ("semicircular canals", "vestibular system", "associated_with"),
        ("vestibular symptoms", "dizziness", "may_produce"),
        ("temporomandibular joint", "vestibular symptoms", "may_contribute_to"),
    ]
    for section_spans in by_section.values():
        section_entities = set()
        for span in section_spans:
            section_entities.update(detected_by_span.get(str(span.get("span_id")), set()))
        for pair in bridge_pairs:
            source, target, rel = pair[:3]
            risk = pair[3] if len(pair) > 3 else "medium"
            if source in section_entities and target in section_entities:
                support_span = next((s for s in section_spans if source in detected_by_span.get(str(s.get("span_id")), set()) or target in detected_by_span.get(str(s.get("span_id")), set())), section_spans[0])
                add_edge(edge_map, source, target, rel, "inferred_from_same_section", "weak", risk, support_span, [source, target], "conservative same-section bridge for mechanism path continuity")

    article_bridge_pairs = [
        ("psoas", "anterior femoral glide", "may_contribute_to"),
        ("posterior pelvic tilt", "anterior femoral glide", "may_contribute_to"),
        ("anterior femoral glide", "femoroacetabular impingement", "may_contribute_to"),
        ("femoroacetabular impingement", "labral impingement", "may_contribute_to"),
        ("labral impingement", "hip pain", "may_produce"),
        ("psoas", "lumbar plexus", "may_compress"),
        ("lumbar plexus", "femoral nerve", "may_contribute_to"),
        ("lumbar plexus", "genitofemoral nerve", "may_contribute_to"),
        ("lumbar plexus", "ilioinguinal nerve", "may_contribute_to"),
        ("lumbar plexus", "lateral femoral cutaneous nerve", "may_contribute_to"),
        ("femoral nerve", "groin pain", "may_produce"),
        ("genitofemoral nerve", "groin pain", "may_produce"),
        ("ilioinguinal nerve", "groin pain", "may_produce"),
        ("posterior pelvic tilt", "disc loading", "may_contribute_to"),
        ("lumbar lordosis", "disc loading", "may_contribute_to"),
        ("disc loading", "disc herniation", "may_contribute_to"),
        ("disc herniation", "nerve root", "may_compress"),
        ("nerve root", "low back pain", "may_produce"),
        ("temporomandibular joint", "eustachian tube", "may_contribute_to"),
        ("eustachian tube", "tinnitus", "may_contribute_to"),
        ("mandibular retraction", "temporomandibular joint", "may_contribute_to"),
        ("temporomandibular joint", "disc displacement", "may_contribute_to"),
        ("disc displacement", "jaw pain", "may_produce"),
        ("vascular TOS", "sympathetic involvement", "may_contribute_to"),
        ("sympathetic involvement", "autonomic symptoms", "may_produce"),
        ("craniovascular hyperperfusion", "autonomic symptoms", "may_contribute_to"),
        ("venous outflow impairment", "ME/CFS", "may_contribute_to"),
        ("intracranial hypertension", "ME/CFS", "may_contribute_to"),
        ("ME/CFS", "fatigue", "associated_with"),
        ("ME/CFS", "brain fog", "associated_with"),
        ("internal jugular vein", "venous outflow impairment", "may_contribute_to"),
        ("venous outflow impairment", "migraine", "may_contribute_to"),
        ("vestibulocochlear nerve", "vestibular system", "associated_with"),
        ("semicircular canals", "vestibular system", "associated_with"),
        ("vestibular system", "vestibular symptoms", "may_produce"),
    ]
    by_article: Dict[str, List[Dict[str, Any]]] = {}
    for span in spans:
        by_article.setdefault(str(span.get("article_id")), []).append(span)
    for article_spans in by_article.values():
        article_entities = set()
        for span in article_spans:
            article_entities.update(detected_by_span.get(str(span.get("span_id")), set()))
        for source, target, rel in article_bridge_pairs:
            if source in article_entities and target in article_entities:
                support_span = next(
                    (
                        s for s in article_spans
                        if source in detected_by_span.get(str(s.get("span_id")), set())
                        or target in detected_by_span.get(str(s.get("span_id")), set())
                    ),
                    article_spans[0],
                )
                add_edge(edge_map, source, target, rel, "inferred_from_path", "weak", "medium", support_span, [source, target], "conservative same-article bridge for audited path continuity")


def build_nodes(spans: List[Dict[str, Any]], sections: List[Dict[str, Any]], articles: List[Dict[str, Any]], detected_by_span: Dict[str, Set[str]]) -> List[Dict[str, Any]]:
    entities = all_entities()
    detected = sorted({entity for values in detected_by_span.values() for entity in values})
    nodes = []
    for name in detected:
        data = entities[name]
        source_span_ids = [str(span.get("span_id")) for span in spans if name in detected_by_span.get(str(span.get("span_id")), set())]
        span_set = set(source_span_ids)
        source_section_ids = sorted({str(span.get("section_id")) for span in spans if str(span.get("span_id")) in span_set and span.get("section_id")})
        source_article_ids = sorted({str(span.get("article_id")) for span in spans if str(span.get("span_id")) in span_set and span.get("article_id")})
        nodes.append(
            {
                "node_id": data["node_id"],
                "canonical_name": name,
                "aliases": data["aliases"],
                "node_type": data["node_type"],
                "description": f"Deterministically detected {data['node_type']} concept from corpus evidence spans.",
                "source_span_ids": source_span_ids,
                "source_section_ids": source_section_ids,
                "source_article_ids": source_article_ids,
                "confidence": round(min(0.95, 0.55 + 0.03 * len(source_span_ids)), 2),
                "schema_version": SCHEMA_VERSION,
            }
        )
    return sorted(nodes, key=lambda n: n["node_id"])


def build_claims(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    claims = []
    relation_claim_type = {
        "innervates": "anatomical_fact",
        "supplies": "anatomical_fact",
        "passes_through": "anatomical_fact",
        "tests_for": "test_claim",
        "red_flag_for": "safety_claim",
    }
    for edge in edges:
        rel = edge["relation_type"]
        claim_type = relation_claim_type.get(rel, "symptom_claim" if rel == "may_produce" else "mechanism_claim")
        claim_text = f"{edge['source_node_id']} {rel} {edge['target_node_id']}"
        claims.append(
            {
                "claim_id": stable_id("claim", [edge["edge_id"], rel]),
                "claim_text": claim_text,
                "claim_type": claim_type,
                "source_span_ids": edge.get("source_span_ids", []),
                "source_section_ids": edge.get("source_section_ids", []),
                "source_article_ids": edge.get("source_article_ids", []),
                "involved_node_ids": [edge["source_node_id"], edge["target_node_id"]],
                "involved_edge_ids": [edge["edge_id"]],
                "support_level": edge.get("support_level", "weak"),
                "clinical_risk": edge.get("clinical_risk", "low"),
                "schema_version": SCHEMA_VERSION,
            }
        )
    return claims


def build_concept_graph(hier_dir: Path = HIER_DIR, graph_dir: Path = GRAPH_DIR) -> Dict[str, Any]:
    spans = read_jsonl(hier_dir / "evidence_spans.jsonl")
    sections = read_jsonl(hier_dir / "sections.jsonl")
    articles = read_jsonl(hier_dir / "articles.jsonl")

    detected_by_span: Dict[str, Set[str]] = {}
    edge_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for span in spans:
        entities = detect_entities(span.get("text", ""))
        detected_by_span[str(span.get("span_id"))] = entities
        if len(entities) >= 2:
            relation_edges_for_span(edge_map, span, entities)
    add_bridge_edges(edge_map, spans, detected_by_span)

    nodes = build_nodes(spans, sections, articles, detected_by_span)
    node_ids = {node["node_id"] for node in nodes}
    edges = sorted(
        [edge for edge in edge_map.values() if edge["source_node_id"] in node_ids and edge["target_node_id"] in node_ids and edge.get("source_span_ids")],
        key=lambda e: e["edge_id"],
    )
    paths = build_mechanism_paths(nodes, edges)
    claims = build_claims(edges)

    graph_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(graph_dir / "nodes.jsonl", nodes)
    write_jsonl(graph_dir / "edges.jsonl", edges)
    write_jsonl(graph_dir / "paths.jsonl", paths)
    write_jsonl(graph_dir / "claims.jsonl", claims)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "builder": "scripts/build_concept_graph.py",
        "source_hierarchical_dir": str(hier_dir.relative_to(PROJECT_ROOT)) if hier_dir.is_relative_to(PROJECT_ROOT) else str(hier_dir),
        "source_counts": {"articles": len(articles), "sections": len(sections), "evidence_spans": len(spans)},
        "graph_counts": {"nodes": len(nodes), "edges": len(edges), "paths": len(paths), "claims": len(claims)},
        "extraction_mode": "deterministic_rules_no_llm",
    }
    (graph_dir / "graph_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    manifest = build_concept_graph()
    print(json.dumps(manifest["graph_counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
