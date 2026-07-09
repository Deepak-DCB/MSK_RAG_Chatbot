# Model Tiering Rules

Pick the smallest capable model for each task while preserving safety and reliability.

## Default tiers
- `openai/gpt-5.3-codex` – orchestrator, retrieval-engineer, evaluation-engineer, clinical-reasoner, safety-auditor, release-manager.
- `openai/gpt-5.1-codex` – backend-api implementation and other moderate-complexity code edits.
- `openai/gpt-5.1-codex-mini` – frontend-integration, librarian, and other mechanical/doc formatting tasks.

## Escalation rules
- Escalate to `gpt-5.3-codex` when uncertainty is high, medical safety is involved, or contracts may drift.
- De-escalate to `gpt-5.1-codex-mini` for deterministic doc or UI tweaks when guidance already exists.

## Cost discipline
- Prefer dry-run evaluation steps before paid runs.
- Document any temporary deviations from this tiering in `docs/opencode/memory/`.
