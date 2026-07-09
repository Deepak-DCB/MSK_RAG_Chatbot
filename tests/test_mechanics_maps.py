from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "VectorDB") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "VectorDB"))

from mechanics_retrieval import (  # noqa: E402
    build_mechanics_context,
    find_entrapment_sites,
    find_mechanism_chains,
    find_muscle_pairs,
    find_nerve_map,
    load_mechanics_maps,
)
from scripts.build_mechanics_maps import build_mechanics_maps  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_build_script_creates_mechanics_artifacts(tmp_path):
    out_dir = tmp_path / "mechanics"
    report_path = tmp_path / "mechanics_map_report.md"
    manifest = build_mechanics_maps(output_dir=out_dir, report_path=report_path)

    required = {
        "nerves.jsonl",
        "entrapment_sites.jsonl",
        "muscles.jsonl",
        "muscle_pairs.jsonl",
        "spaces.jsonl",
        "mechanism_chains.jsonl",
        "mechanics_manifest.json",
    }
    assert required <= {path.name for path in out_dir.iterdir()}
    assert report_path.exists()
    assert manifest["extraction_mode"] == "deterministic_pilot_no_llm_no_external_sources"


def test_mechanics_records_have_required_fields(tmp_path):
    out_dir = tmp_path / "mechanics"
    build_mechanics_maps(output_dir=out_dir, report_path=tmp_path / "report.md")

    nerve_fields = {
        "nerve_id",
        "name",
        "aliases",
        "parent_plexus_or_root_if_supported",
        "course_summary",
        "entrapment_site_ids",
        "symptom_ids",
        "evidence_span_ids",
        "support_level",
        "notes",
    }
    site_fields = {
        "site_id",
        "nerve_id",
        "site_name",
        "anatomical_region",
        "nearby_muscles",
        "nearby_bones_or_joints",
        "nearby_spaces",
        "mechanical_trigger",
        "symptoms",
        "tests_or_assessments",
        "exercise_or_posture_implications",
        "direct_support_span_ids",
        "indirect_support_span_ids",
        "unsupported_or_uncertain_notes",
        "support_level",
    }
    pair_fields = {
        "pair_id",
        "muscles",
        "region",
        "relationship_type",
        "mechanical_role",
        "space_or_structure_affected",
        "related_nerves_or_vessels",
        "related_symptoms",
        "evidence_span_ids",
        "support_level",
        "notes",
    }
    chain_fields = {
        "chain_id",
        "question_it_answers",
        "steps",
        "involved_structures",
        "involved_muscles",
        "involved_nerves_or_vessels",
        "weakest_step",
        "direct_support_span_ids",
        "indirect_support_span_ids",
        "support_level",
        "safety_boundary",
    }

    assert all(nerve_fields <= record.keys() for record in read_jsonl(out_dir / "nerves.jsonl"))
    assert all(site_fields <= record.keys() for record in read_jsonl(out_dir / "entrapment_sites.jsonl"))
    assert all(pair_fields <= record.keys() for record in read_jsonl(out_dir / "muscle_pairs.jsonl"))
    assert all(chain_fields <= record.keys() for record in read_jsonl(out_dir / "mechanism_chains.jsonl"))


def test_pilot_records_exist_and_supported_pairs_have_spans(tmp_path):
    out_dir = tmp_path / "mechanics"
    build_mechanics_maps(output_dir=out_dir, report_path=tmp_path / "report.md")
    nerves = read_jsonl(out_dir / "nerves.jsonl")
    pairs = read_jsonl(out_dir / "muscle_pairs.jsonl")
    chains = read_jsonl(out_dir / "mechanism_chains.jsonl")

    nerve_names = {record["name"] for record in nerves}
    assert {"dorsal scapular nerve", "brachial plexus"} <= nerve_names
    for pair in pairs:
        if pair["support_level"] != "unsupported":
            assert pair["evidence_span_ids"]
    assert all(chain.get("weakest_step") for chain in chains)


def test_mechanics_retrieval_returns_relevant_pilot_maps(tmp_path):
    out_dir = tmp_path / "mechanics"
    build_mechanics_maps(output_dir=out_dir, report_path=tmp_path / "report.md")
    maps = load_mechanics_maps(out_dir)

    dsn_nerves = find_nerve_map("where can dorsal scapular nerve be entrapped", maps)
    dsn_sites = find_entrapment_sites("where can dorsal scapular nerve be entrapped", maps)
    assert any(record["nerve_id"] == "nerve_dorsal_scapular_nerve" for record in dsn_nerves)
    assert any(record["nerve_id"] == "nerve_dorsal_scapular_nerve" for record in dsn_sites)

    pairs = find_muscle_pairs("how do traps and scalenes work together", maps)
    chains = find_mechanism_chains("how do traps and scalenes work together", maps)
    assert any("trapezius" in record["muscles"] and "scalenes" in record["muscles"] for record in pairs)
    assert any("trapezius" in record["involved_muscles"] and "scalenes" in record["involved_muscles"] for record in chains)

    sites = find_entrapment_sites("costoclavicular compression brachial plexus", maps)
    assert any(record["site_id"] == "site_brachial_plexus_costoclavicular_passage" for record in sites)

    context = build_mechanics_context("costoclavicular compression brachial plexus", base_dir=out_dir)
    assert context["available"] is True
    assert "costoclavicular" in context["context"].lower()
