from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HIERARCHY_DIR = PROJECT_ROOT / "MSKArticlesINDEX" / "hierarchical"
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "MSKArticlesINDEX" / "graph"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "MSKArticlesINDEX" / "mechanics"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "Evaluation" / "mechanics_map_report.md"
SCHEMA_VERSION = "1.0.0"


PILOT_SCOPE = [
    "thoracic outlet / scapular mechanics",
    "dorsal scapular nerve",
    "brachial plexus",
    "scalenes",
    "trapezius",
    "levator scapulae",
    "rhomboids",
    "first rib",
    "clavicle",
    "costoclavicular space",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(record, sort_keys=True) for record in records)
    path.write_text(f"{text}\n" if text else "", encoding="utf-8")


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def span_ids_for_terms(spans: list[dict[str, Any]], required_terms: list[str], limit: int = 8) -> list[str]:
    ids: list[str] = []
    for span in spans:
        text = span.get("text", "").lower()
        if all(term.lower() in text for term in required_terms):
            ids.append(span.get("span_id", ""))
        if len(ids) >= limit:
            break
    return unique(ids)


def known_span_ids(spans: list[dict[str, Any]], ids: list[str]) -> list[str]:
    available = {span.get("span_id") for span in spans}
    return [span_id for span_id in ids if span_id in available]


def spans_for_graph_node(nodes: list[dict[str, Any]], canonical_name: str, limit: int = 8) -> list[str]:
    for node in nodes:
        if node.get("canonical_name") == canonical_name:
            return list(node.get("source_span_ids", []))[:limit]
    return []


def path_support(paths: list[dict[str, Any]], text_fragment: str) -> tuple[list[str], str]:
    for path in paths:
        if text_fragment.lower() in path.get("path_text", "").lower():
            return list(path.get("source_span_ids", [])), path.get("path_support", "indirect")
    return [], "unsupported"


def edge_support(
    edges: list[dict[str, Any]], source_node_id: str, target_node_id: str, limit: int = 8
) -> tuple[list[str], str]:
    span_ids: list[str] = []
    levels: list[str] = []
    for edge in edges:
        if edge.get("source_node_id") == source_node_id and edge.get("target_node_id") == target_node_id:
            span_ids.extend(edge.get("source_span_ids", []))
            levels.append(edge.get("support_level", "weak"))
    if not span_ids:
        return [], "unsupported"
    order = {"direct": 0, "indirect": 1, "weak": 2, "unsupported": 3}
    support = sorted(levels, key=lambda level: order.get(level, 3))[0]
    return unique(span_ids)[:limit], support


def make_nerve_records(spans: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dsn_spans = unique(
        known_span_ids(spans, ["67c0c170eb3517fcf9c82dd275f4e709", "bce1016b5782bdd120224975c2159999"])
        + spans_for_graph_node(nodes, "dorsal scapular nerve", 6)
    )
    brachial_spans = unique(
        known_span_ids(
            spans,
            [
                "7110fad0fa003d595956ef8910d7f95e",
                "7a06cd61962fcf1221e0f3c5b4a8f6b2",
                "d2f6b9f0cb3d9601de622105d3a0ba06",
                "cd05cb4fc90b0bf70527eb83ab0e7be1",
            ],
        )
        + spans_for_graph_node(nodes, "brachial plexus", 8)
    )
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "nerve_id": "nerve_dorsal_scapular_nerve",
            "name": "dorsal scapular nerve",
            "aliases": ["dorsal scapular nerve", "DSN"],
            "parent_plexus_or_root_if_supported": "unsupported in pilot spans",
            "course_summary": "Pilot evidence supports pain between the shoulder blades via the dorsal scapular nerve, but does not yet map a full anatomical course.",
            "entrapment_site_ids": ["site_dsn_thoracic_outlet_uncertain"],
            "symptom_ids": ["pain_between_shoulder_blades", "neck_pain", "dorsal_scapular_pain"],
            "evidence_span_ids": dsn_spans,
            "support_level": "indirect" if dsn_spans else "unsupported",
            "notes": "Site specificity remains uncertain; this pilot does not assert scalene, levator scapulae, or rhomboid entrapment for DSN without direct span support.",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "nerve_id": "nerve_brachial_plexus",
            "name": "brachial plexus",
            "aliases": ["brachial plexus", "brachial plexus nerve network"],
            "parent_plexus_or_root_if_supported": "C8 and T1 roots are described as susceptible in the costoclavicular interval; full root map not completed in this pilot.",
            "course_summary": "Pilot evidence describes the brachial plexus passing through thoracic outlet spaces, especially the interscalene triangle and costoclavicular passage.",
            "entrapment_site_ids": ["site_brachial_plexus_interscalene_triangle", "site_brachial_plexus_costoclavicular_passage"],
            "symptom_ids": ["paresthesia", "ulnar_neuralgia", "arm_pain", "neck_pain"],
            "evidence_span_ids": brachial_spans,
            "support_level": "direct" if brachial_spans else "unsupported",
            "notes": "This record is limited to the thoracic outlet/scapular pilot and does not attempt full brachial plexus branch completion.",
        },
    ]


def make_entrapment_sites(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    interscalene = known_span_ids(
        spans,
        ["d2f6b9f0cb3d9601de622105d3a0ba06", "cd05cb4fc90b0bf70527eb83ab0e7be1", "1cdfdc9e32aea34948c9bc91576f2eb9"],
    )
    costoclavicular = known_span_ids(
        spans,
        [
            "26c37d6dcc47097ba4d1c1169b2baa92",
            "8fb432711bf9571a7aa6bc8c0c3f660e",
            "f561444e10083135f908bff1aabb4e5d",
            "e945c7b3e379384abcf269f2abe821c7",
            "f7f97853b7ca791b4dc149e756dc9324",
            "cf9b9879695b5f19f2cafa5df3c738ea",
        ],
    )
    dsn = known_span_ids(spans, ["67c0c170eb3517fcf9c82dd275f4e709", "bce1016b5782bdd120224975c2159999"])
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "site_id": "site_brachial_plexus_interscalene_triangle",
            "nerve_id": "nerve_brachial_plexus",
            "site_name": "interscalene triangle",
            "anatomical_region": "thoracic outlet / lateral neck",
            "nearby_muscles": ["anterior scalene", "middle scalene", "scalenes"],
            "nearby_bones_or_joints": ["first rib"],
            "nearby_spaces": ["thoracic outlet", "interscalene triangle"],
            "mechanical_trigger": "Scalene inhibition/tightness and pressure into the distal anterior/middle scalene region are described as relevant to brachial plexus irritation.",
            "symptoms": ["arm pain", "paresthesia", "neck pain", "scapular pain"],
            "tests_or_assessments": ["pressure into the brachial plexus between distal anterior and middle scalene fibers", "Morley's test"],
            "exercise_or_posture_implications": ["start scalene loading carefully if used", "avoid overloading irritable symptoms"],
            "direct_support_span_ids": interscalene,
            "indirect_support_span_ids": [],
            "unsupported_or_uncertain_notes": "Does not prove all symptoms are caused by this site; testing language comes from corpus and is not a diagnostic instruction.",
            "support_level": "direct" if interscalene else "unsupported",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "site_id": "site_brachial_plexus_costoclavicular_passage",
            "nerve_id": "nerve_brachial_plexus",
            "site_name": "costoclavicular passage",
            "anatomical_region": "thoracic outlet / clavicle-first rib interval",
            "nearby_muscles": ["scalenes", "trapezius", "pectoralis minor"],
            "nearby_bones_or_joints": ["clavicle", "first rib", "scapula"],
            "nearby_spaces": ["costoclavicular space", "thoracic outlet"],
            "mechanical_trigger": "Scalene-related first-rib elevation and depressed clavicle/scapular position are described as narrowing or compressing the costoclavicular region.",
            "symptoms": ["tingling", "arm pain", "hand pain", "chest pain", "neck pain", "scapular pain"],
            "tests_or_assessments": ["clavicular depression test"],
            "exercise_or_posture_implications": ["avoid back-and-down shoulder cueing in this model", "slight shoulder elevation is described as decompressing the thoracic outlet"],
            "direct_support_span_ids": costoclavicular,
            "indirect_support_span_ids": [],
            "unsupported_or_uncertain_notes": "The corpus discusses provocative tests and posture effects; this artifact does not prescribe treatment or establish diagnosis.",
            "support_level": "direct" if costoclavicular else "unsupported",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "site_id": "site_dsn_thoracic_outlet_uncertain",
            "nerve_id": "nerve_dorsal_scapular_nerve",
            "site_name": "thoracic outlet / periscapular relationship, site not localized",
            "anatomical_region": "thoracic outlet and periscapular region",
            "nearby_muscles": [],
            "nearby_bones_or_joints": [],
            "nearby_spaces": ["thoracic outlet"],
            "mechanical_trigger": "Unsupported in pilot as a localized entrapment site; evidence supports symptom association more than site anatomy.",
            "symptoms": ["pain between shoulder blades", "neck pain", "periscapular pain"],
            "tests_or_assessments": [],
            "exercise_or_posture_implications": [],
            "direct_support_span_ids": [],
            "indirect_support_span_ids": dsn,
            "unsupported_or_uncertain_notes": "No pilot span directly proves a DSN entrapment site at scalenes, levator scapulae, or rhomboids.",
            "support_level": "indirect" if dsn else "unsupported",
        },
    ]


def make_muscle_records(spans: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("muscle_scalenes", "scalenes", ["scalenes", "scalene", "anterior scalene", "middle scalene"], "lateral neck / thoracic outlet"),
        ("muscle_trapezius", "trapezius", ["trapezius", "trapezius muscle"], "neck / scapular girdle"),
        ("muscle_levator_scapulae", "levator scapulae", ["levator scapulae", "LS"], "neck / scapular girdle"),
        ("muscle_rhomboids", "rhomboids", ["rhomboids", "rhomboid"], "scapular girdle"),
    ]
    records: list[dict[str, Any]] = []
    for muscle_id, name, aliases, region in specs:
        span_ids = spans_for_graph_node(nodes, name, 8) or span_ids_for_terms(spans, [name.split()[0]], 8)
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "muscle_id": muscle_id,
                "name": name,
                "aliases": aliases,
                "region": region,
                "mechanical_roles": [],
                "related_spaces": [],
                "related_nerves_or_vessels": [],
                "evidence_span_ids": span_ids,
                "support_level": "indirect" if span_ids else "unsupported",
                "notes": "Pilot muscle node seeded from graph/hierarchy evidence; role-specific claims are captured in muscle_pairs and mechanism_chains.",
            }
        )
    return records


def make_muscle_pairs(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scalene_trap = known_span_ids(spans, ["734a885ca001f9efa1309d2a978d98ea", "f40c9ea13e18134833c12403dd5cb546"])
    trap_levator = known_span_ids(spans, ["c29ee2ec6d444b0a4f82ccb37e1d52dc", "e4e9801b4e9bf1c9c88dcaa163774966", "f9ee504978c2b7bee456e72c5d0c7c82"])
    scalene_first_rib = known_span_ids(spans, ["26c37d6dcc47097ba4d1c1169b2baa92", "eb0d5832808c8c292a35f7a3e3498c5b"])
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "pair_id": "pair_scalenes_trapezius_breathing_clavicle",
            "muscles": ["scalenes", "trapezius"],
            "region": "thoracic outlet / shoulder girdle",
            "relationship_type": "synchronized breathing and clavicle/scapula support",
            "mechanical_role": "Corpus describes ribs and clavicle elevating slightly during inspiration in synchronization by scalenes, trapezius, and other muscles.",
            "space_or_structure_affected": ["clavicle", "ribs", "thoracic outlet"],
            "related_nerves_or_vessels": ["brachial plexus", "subclavian vessels"],
            "related_symptoms": [],
            "evidence_span_ids": scalene_trap,
            "support_level": "direct" if scalene_trap else "unsupported",
            "notes": "This is a study-map relationship, not proof that the pair alone causes or resolves symptoms.",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "pair_id": "pair_trapezius_levator_scapulae_scapular_resting",
            "muscles": ["trapezius", "levator scapulae"],
            "region": "scapular girdle / neck",
            "relationship_type": "scapular resting height and postural support",
            "mechanical_role": "Corpus links low scapular resting position with trapezius, levator scapulae, and scalene inhibition/tightness, and separately discusses trapezius and levator strengthening.",
            "space_or_structure_affected": ["scapula", "neck", "thoracic outlet"],
            "related_nerves_or_vessels": [],
            "related_symptoms": ["neck pain", "shoulder pain"],
            "evidence_span_ids": trap_levator,
            "support_level": "indirect" if trap_levator else "unsupported",
            "notes": "The pilot records co-mechanical framing only; it does not claim a direct nerve entrapment pathway for levator scapulae or rhomboids.",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "pair_id": "pair_scalenes_first_rib_costoclavicular",
            "muscles": ["scalenes"],
            "region": "thoracic outlet",
            "relationship_type": "muscle-to-bone space influence",
            "mechanical_role": "Corpus states scalene attachment to ribs may elevate the first rib and increase secondary compression potential between first rib and clavicle.",
            "space_or_structure_affected": ["first rib", "clavicle", "costoclavicular space"],
            "related_nerves_or_vessels": ["brachial plexus"],
            "related_symptoms": ["paresthesia", "arm pain"],
            "evidence_span_ids": scalene_first_rib,
            "support_level": "direct" if scalene_first_rib else "unsupported",
            "notes": "Recorded as a narrow corpus-supported mechanism fragment.",
        },
    ]


def make_spaces(spans: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("space_interscalene_triangle", "interscalene triangle", ["d2f6b9f0cb3d9601de622105d3a0ba06", "c9124f158144d45527a2bc224013841d"]),
        ("space_costoclavicular_space", "costoclavicular space", ["e945c7b3e379384abcf269f2abe821c7", "f561444e10083135f908bff1aabb4e5d", "26c37d6dcc47097ba4d1c1169b2baa92"]),
        ("space_thoracic_outlet", "thoracic outlet", ["7a06cd61962fcf1221e0f3c5b4a8f6b2", "2b91619e4b74d78b5a577fc52cd288f4"]),
    ]
    records = []
    for space_id, name, seed_ids in specs:
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "space_id": space_id,
                "name": name,
                "aliases": [name],
                "region": "thoracic outlet / scapular pilot",
                "nearby_structures": [],
                "related_nerves_or_vessels": ["brachial plexus"] if name != "thoracic outlet" else ["brachial plexus", "subclavian vessels"],
                "evidence_span_ids": unique(known_span_ids(spans, seed_ids) + spans_for_graph_node(nodes, name, 6)),
                "support_level": "direct",
                "notes": "Pilot space record; nearby structures are detailed in entrapment site records when directly supported.",
            }
        )
    return records


def make_mechanism_chains(spans: list[dict[str, Any]], paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scalene_path_spans, scalene_path_support = path_support(paths, "scalene -> first rib -> costoclavicular space -> brachial plexus")
    scap_path_spans, scap_path_support = path_support(paths, "scapular depression -> clavicle -> costoclavicular space -> brachial plexus")
    dsn_path_spans, dsn_path_support = path_support(paths, "thoracic outlet syndrome -> dorsal scapular nerve")
    trap_spans = known_span_ids(spans, ["734a885ca001f9efa1309d2a978d98ea", "39641f9f42a417c6fa7ec6d688b7580d"])
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "chain_id": "chain_scalenes_first_rib_costoclavicular_brachial_plexus",
            "question_it_answers": "How can scalene mechanics change first-rib/costoclavicular space and brachial plexus load?",
            "steps": [
                "scalenes attach to/elevate ribs during inspiration",
                "scalene tightness may elevate the first rib",
                "first rib elevation can increase compression potential between first rib and clavicle",
                "costoclavicular compression can affect the brachial plexus",
            ],
            "involved_structures": ["first rib", "clavicle", "costoclavicular space", "thoracic outlet"],
            "involved_muscles": ["scalenes"],
            "involved_nerves_or_vessels": ["brachial plexus", "subclavian vessels"],
            "weakest_step": "Graph path is indirect and should be explained as a possible mechanism, not a diagnosis.",
            "direct_support_span_ids": known_span_ids(spans, ["26c37d6dcc47097ba4d1c1169b2baa92", "8fb432711bf9571a7aa6bc8c0c3f660e"]),
            "indirect_support_span_ids": scalene_path_spans,
            "support_level": scalene_path_support,
            "safety_boundary": "Do not infer that this mechanism is present in a specific person without clinical evaluation; escalate progressive neurologic symptoms.",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "chain_id": "chain_scapular_depression_clavicle_brachial_plexus",
            "question_it_answers": "How can scapular depression or back-and-down posture affect costoclavicular compression?",
            "steps": [
                "scapular depression/anterior tilt can lower or alter clavicle position",
                "clavicle may descend toward the first rib/costoclavicular space",
                "the corpus describes brachial plexus and vessel compression in that setting",
            ],
            "involved_structures": ["scapula", "clavicle", "first rib", "costoclavicular space"],
            "involved_muscles": ["trapezius", "levator scapulae", "scalenes"],
            "involved_nerves_or_vessels": ["brachial plexus", "subclavian vessels"],
            "weakest_step": "Specific symptom attribution remains indirect; posture-space relationship is stronger than individual diagnosis.",
            "direct_support_span_ids": known_span_ids(spans, ["f561444e10083135f908bff1aabb4e5d", "e945c7b3e379384abcf269f2abe821c7", "8a44fb8bc0f8db100e7111fc25cf916d"]),
            "indirect_support_span_ids": scap_path_spans,
            "support_level": scap_path_support,
            "safety_boundary": "Avoid presenting posture correction as treatment prescription; use as study context only.",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "chain_id": "chain_dorsal_scapular_nerve_tos_pain_uncertain_site",
            "question_it_answers": "What does the corpus support about dorsal scapular nerve and interscapular pain?",
            "steps": [
                "TOS/neurogenic context mentions dorsal scapular nerve",
                "pain between shoulder blades is described via the dorsal scapular nerve",
                "neck pain is also mentioned in the same evidence span",
            ],
            "involved_structures": ["thoracic outlet", "periscapular region"],
            "involved_muscles": [],
            "involved_nerves_or_vessels": ["dorsal scapular nerve"],
            "weakest_step": "Entrapment site is not localized in pilot evidence.",
            "direct_support_span_ids": [],
            "indirect_support_span_ids": unique(known_span_ids(spans, ["67c0c170eb3517fcf9c82dd275f4e709", "bce1016b5782bdd120224975c2159999"]) + dsn_path_spans),
            "support_level": dsn_path_support if dsn_path_support != "unsupported" else "indirect",
            "safety_boundary": "Do not claim DSN entrapment site, diagnosis, or causal certainty from this pilot record.",
        },
        {
            "schema_version": SCHEMA_VERSION,
            "chain_id": "chain_scalenes_trapezius_breathing_clavicle",
            "question_it_answers": "How do traps and scalenes work together in this corpus model?",
            "steps": [
                "scalenes elevate ribs during inspiration",
                "the corpus describes ribs and clavicle elevating slightly in synchronization by scalenes, trapezius, and other muscles",
                "trapezius supports scapula/clavicle position in the thoracic outlet model",
            ],
            "involved_structures": ["ribs", "clavicle", "scapula", "thoracic outlet"],
            "involved_muscles": ["scalenes", "trapezius"],
            "involved_nerves_or_vessels": ["brachial plexus"],
            "weakest_step": "Synchronization is directly described, but downstream nerve-load effect is indirect.",
            "direct_support_span_ids": known_span_ids(spans, ["734a885ca001f9efa1309d2a978d98ea"]),
            "indirect_support_span_ids": trap_spans,
            "support_level": "indirect" if trap_spans else "unsupported",
            "safety_boundary": "Study map only; not an exercise prescription.",
        },
    ]


def write_manifest(output_dir: Path, records_by_file: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts = {name: len(records) for name, records in records_by_file.items()}
    support_counts = Counter(
        record.get("support_level", "unknown") for records in records_by_file.values() for record in records
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "builder": "scripts/build_mechanics_maps.py",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "extraction_mode": "deterministic_pilot_no_llm_no_external_sources",
        "pilot_scope": PILOT_SCOPE,
        "record_counts": counts,
        "support_counts": dict(sorted(support_counts.items())),
        "inputs": {
            "evidence_spans": "MSKArticlesINDEX/hierarchical/evidence_spans.jsonl",
            "sections": "MSKArticlesINDEX/hierarchical/sections.jsonl",
            "graph_nodes": "MSKArticlesINDEX/graph/nodes.jsonl",
            "graph_edges": "MSKArticlesINDEX/graph/edges.jsonl",
            "graph_paths": "MSKArticlesINDEX/graph/paths.jsonl",
        },
        "limitations": [
            "Pilot scope only; not all nerves or body regions are mapped.",
            "Records are corpus-study aids, not diagnostic claims or treatment recommendations.",
            "Empty fields and unsupported notes are intentional when pilot evidence does not support a claim.",
        ],
    }
    (output_dir / "mechanics_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_report(report_path: Path, manifest: dict[str, Any], records_by_file: dict[str, list[dict[str, Any]]]) -> None:
    nerves = records_by_file["nerves.jsonl"]
    sites = records_by_file["entrapment_sites.jsonl"]
    pairs = records_by_file["muscle_pairs.jsonl"]
    chains = records_by_file["mechanism_chains.jsonl"]
    strongest_sites = [site for site in sites if site.get("support_level") in {"direct", "indirect"}]
    strongest_pairs = [pair for pair in pairs if pair.get("support_level") in {"direct", "indirect"}]
    unsupported_notes = [site for site in sites if site.get("unsupported_or_uncertain_notes")]
    lines = [
        "# Mechanics Map Report",
        "",
        "This developer-facing report summarizes the deterministic Body Mechanics Study Map pilot. It is not runtime answer behavior and does not add external medical sources.",
        "",
        "## Records Created",
    ]
    for name, count in manifest["record_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Pilot Coverage"])
    for item in manifest["pilot_scope"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Strongest Supported Entrapment Sites"])
    for site in strongest_sites:
        lines.append(f"- `{site['site_id']}`: {site['site_name']} ({site['support_level']}) spans={site['direct_support_span_ids'] or site['indirect_support_span_ids']}")
    lines.extend(["", "## Strongest Muscle-Pair Relationships"])
    for pair in strongest_pairs:
        lines.append(f"- `{pair['pair_id']}`: {', '.join(pair['muscles'])} ({pair['support_level']}) spans={pair['evidence_span_ids']}")
    lines.extend(["", "## Unsupported Or Uncertain Relationships"])
    for site in unsupported_notes:
        lines.append(f"- `{site['site_id']}`: {site['unsupported_or_uncertain_notes']}")
    lines.extend(["", "## Examples"])
    examples = [
        ("Dorsal scapular nerve", nerves[0]),
        ("Brachial plexus", nerves[1]),
        ("Scalenes + trapezius", pairs[0]),
        ("Costoclavicular space", sites[1]),
        ("Mechanism chain", chains[0]),
    ]
    for label, record in examples:
        lines.append(f"- {label}: `{record.get('nerve_id') or record.get('pair_id') or record.get('site_id') or record.get('chain_id')}` support={record.get('support_level')}")
    lines.extend(
        [
            "",
            "## Limits",
            "- The pilot does not complete every nerve.",
            "- Dorsal scapular nerve site localization remains unsupported/uncertain in the pilot.",
            "- Mechanism chains distinguish direct spans from indirect graph/path support and must not be presented as diagnosis.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_mechanics_maps(
    hierarchy_dir: Path = DEFAULT_HIERARCHY_DIR,
    graph_dir: Path = DEFAULT_GRAPH_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    spans = read_jsonl(hierarchy_dir / "evidence_spans.jsonl")
    read_jsonl(hierarchy_dir / "sections.jsonl")
    nodes = read_jsonl(graph_dir / "nodes.jsonl")
    edges = read_jsonl(graph_dir / "edges.jsonl")
    paths = read_jsonl(graph_dir / "paths.jsonl")

    records_by_file = {
        "nerves.jsonl": make_nerve_records(spans, nodes),
        "entrapment_sites.jsonl": make_entrapment_sites(spans),
        "muscles.jsonl": make_muscle_records(spans, nodes),
        "muscle_pairs.jsonl": make_muscle_pairs(spans),
        "spaces.jsonl": make_spaces(spans, nodes),
        "mechanism_chains.jsonl": make_mechanism_chains(spans, paths),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in records_by_file.items():
        write_jsonl(output_dir / filename, records)
    manifest = write_manifest(output_dir, records_by_file)
    write_report(report_path, manifest, records_by_file)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic Body Mechanics Study Map artifacts.")
    parser.add_argument("--hierarchy-dir", type=Path, default=DEFAULT_HIERARCHY_DIR)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    manifest = build_mechanics_maps(args.hierarchy_dir, args.graph_dir, args.output_dir, args.report_path)
    counts = ", ".join(f"{name}={count}" for name, count in manifest["record_counts"].items())
    print(f"Mechanics maps built: {counts}")


if __name__ == "__main__":
    main()
