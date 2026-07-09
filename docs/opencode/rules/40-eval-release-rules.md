# Evaluation and Release Rules

## Evaluation process
1. Run dry-run validation first.
2. Run bounded paid evaluation with explicit case and cost limits.
3. Save artifacts under `Evaluation/runs/<run_id>/`.

## Artifact requirements
- `cases.jsonl`
- `run_report.json`
- `run_notes.md`

Each run must include reproducibility metadata:
- commit hash
- dataset version and hash
- pipeline mode and key config values

## Reporting discipline
- Distinguish measured metrics from not-evaluated layers.
- Do not report placeholder zeros as measured results.

## Release gates
- Safety and grounding regressions block release.
- Retrieval and reliability regressions require explicit sign-off.
