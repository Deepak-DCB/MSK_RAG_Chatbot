import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_answer_modes import main


QUESTION = "What is the chain from scalenes to first rib to costoclavicular compression?"


def test_dry_run_comparison_creates_reports_without_openai_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    md_path = tmp_path / "comparison.md"
    json_path = tmp_path / "comparison.json"

    rc = main(
        [
            "--question",
            QUESTION,
            "--dry-run",
            "--modes",
            "normal_ask_default,mechanics_study",
            "--out",
            str(md_path),
            "--json-out",
            str(json_path),
        ]
    )

    assert rc == 0
    assert md_path.exists()
    assert json_path.exists()

    markdown = md_path.read_text(encoding="utf-8")
    assert "# Answer Mode Comparison" in markdown
    assert "## Summary Table" in markdown
    assert "### Retrieval process" in markdown
    assert "### Selected information" in markdown
    assert "### Final answer" in markdown
    assert "## Cross-mode comparison" in markdown

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["live_openai"] is False
    assert [result["mode"] for result in data["results"]] == ["normal_ask_default", "mechanics_study"]


def test_mechanics_study_mode_includes_mechanics_metadata_if_artifacts_exist(tmp_path):
    md_path = tmp_path / "mechanics.md"
    json_path = tmp_path / "mechanics.json"

    rc = main(
        [
            "--question",
            QUESTION,
            "--dry-run",
            "--modes",
            "mechanics_study",
            "--out",
            str(md_path),
            "--json-out",
            str(json_path),
        ]
    )

    assert rc == 0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    mechanics = data["results"][0]
    records = mechanics["selected_mechanics_records"]
    record_count = sum(len(records[key]) for key in records)
    if Path("MSKArticlesINDEX/mechanics/mechanics_manifest.json").exists():
        assert record_count > 0
        assert records["mechanism_chains"]
        assert mechanics["selected_evidence_spans"]


def test_live_openai_missing_api_key_fails_gracefully(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rc = main(
        [
            "--question",
            QUESTION,
            "--live-openai",
            "--modes",
            "normal_ask_default",
            "--out",
            str(tmp_path / "live.md"),
            "--json-out",
            str(tmp_path / "live.json"),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert "OPENAI_API_KEY is not set" in captured.err
    assert "Traceback" not in captured.err
