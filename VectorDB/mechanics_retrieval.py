from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MECHANICS_DIR = PROJECT_ROOT / "MSKArticlesINDEX" / "mechanics"

FILES = {
    "nerves": "nerves.jsonl",
    "entrapment_sites": "entrapment_sites.jsonl",
    "muscles": "muscles.jsonl",
    "muscle_pairs": "muscle_pairs.jsonl",
    "spaces": "spaces.jsonl",
    "mechanism_chains": "mechanism_chains.jsonl",
}

QUERY_ALIASES = {
    "traps": "trapezius",
    "trap": "trapezius",
    "dsn": "dorsal scapular nerve",
    "costoclavicular compression": "costoclavicular space compression",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_mechanics_maps(base_dir: Path | str = DEFAULT_MECHANICS_DIR) -> dict[str, Any]:
    base = Path(base_dir)
    maps = {key: _read_jsonl(base / filename) for key, filename in FILES.items()}
    manifest_path = base / "mechanics_manifest.json"
    maps["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    maps["available"] = all((base / filename).exists() for filename in FILES.values())
    maps["base_dir"] = str(base)
    return maps


def _normalize_query(query: str) -> str:
    normalized = query.lower()
    for alias, replacement in QUERY_ALIASES.items():
        normalized = normalized.replace(alias, replacement)
    return normalized


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", _normalize_query(text)) if len(token) > 2}


def _record_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for value in record.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
    return " ".join(parts)


def _rank(query: str, records: list[dict[str, Any]], max_items: int | None = None) -> list[dict[str, Any]]:
    query_tokens = _tokens(query)
    scored: list[tuple[int, dict[str, Any]]] = []
    for record in records:
        text = _record_text(record)
        tokens = _tokens(text)
        score = len(query_tokens & tokens)
        exact_boost = sum(2 for token in query_tokens if token in text.lower())
        score += exact_boost
        if score > 0:
            scored.append((score, record))
    scored.sort(key=lambda item: (-item[0], _record_text(item[1])))
    ranked = [record for _, record in scored]
    return ranked[:max_items] if max_items is not None else ranked


def find_nerve_map(query: str, maps: dict[str, Any] | None = None, max_items: int = 5) -> list[dict[str, Any]]:
    data = maps or load_mechanics_maps()
    return _rank(query, data.get("nerves", []), max_items)


def find_entrapment_sites(query: str, maps: dict[str, Any] | None = None, max_items: int = 5) -> list[dict[str, Any]]:
    data = maps or load_mechanics_maps()
    return _rank(query, data.get("entrapment_sites", []), max_items)


def find_muscle_pairs(query: str, maps: dict[str, Any] | None = None, max_items: int = 5) -> list[dict[str, Any]]:
    data = maps or load_mechanics_maps()
    return _rank(query, data.get("muscle_pairs", []), max_items)


def find_spaces(query: str, maps: dict[str, Any] | None = None, max_items: int = 5) -> list[dict[str, Any]]:
    data = maps or load_mechanics_maps()
    return _rank(query, data.get("spaces", []), max_items)


def find_mechanism_chains(query: str, maps: dict[str, Any] | None = None, max_items: int = 5) -> list[dict[str, Any]]:
    data = maps or load_mechanics_maps()
    return _rank(query, data.get("mechanism_chains", []), max_items)


def build_mechanics_context(query: str, max_items: int = 8, base_dir: Path | str = DEFAULT_MECHANICS_DIR) -> dict[str, Any]:
    maps = load_mechanics_maps(base_dir)
    if not maps.get("available"):
        return {
            "available": False,
            "fallback_reason": "mechanics_artifacts_missing",
            "nerves": [],
            "entrapment_sites": [],
            "muscle_pairs": [],
            "spaces": [],
            "mechanism_chains": [],
            "context": "",
        }

    nerves = find_nerve_map(query, maps, max_items=max_items)
    sites = find_entrapment_sites(query, maps, max_items=max_items)
    pairs = find_muscle_pairs(query, maps, max_items=max_items)
    spaces = find_spaces(query, maps, max_items=max_items)
    chains = find_mechanism_chains(query, maps, max_items=max_items)
    selected: list[str] = []
    for label, records in [
        ("Nerve", nerves),
        ("Entrapment site", sites),
        ("Muscle pair", pairs),
        ("Space", spaces),
        ("Mechanism chain", chains),
    ]:
        for record in records[:max_items]:
            record_id = record.get("nerve_id") or record.get("site_id") or record.get("pair_id") or record.get("space_id") or record.get("chain_id")
            support = record.get("support_level", "unknown")
            summary = record.get("course_summary") or record.get("mechanical_trigger") or record.get("mechanical_role") or record.get("question_it_answers") or ""
            selected.append(f"{label} {record_id} ({support}): {summary}")
            if len(selected) >= max_items:
                break
        if len(selected) >= max_items:
            break

    return {
        "available": True,
        "fallback_reason": "",
        "nerves": nerves[:max_items],
        "entrapment_sites": sites[:max_items],
        "muscle_pairs": pairs[:max_items],
        "spaces": spaces[:max_items],
        "mechanism_chains": chains[:max_items],
        "context": "\n".join(selected),
    }
