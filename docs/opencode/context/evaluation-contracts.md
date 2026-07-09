# Evaluation Contracts

## Primary runner
- `scripts/run_eval_production.py`

## Artifact location
- `Evaluation/runs/<run_id>/cases.jsonl`
- `Evaluation/runs/<run_id>/run_report.json`
- `Evaluation/runs/<run_id>/run_notes.md`

## Schema references
- `Evaluation/automation_eval_schema_v2.json`
- `Evaluation/clinician_review_rubric.txt`

## Required run metadata
- commit hash
- pipeline mode
- model configuration
- dataset path, version, and hash

## Phase discipline
- Use dry-run before paid calls.
- Mark non-measured domains as `not_evaluated`.
- Use bounded runs before full runs.
