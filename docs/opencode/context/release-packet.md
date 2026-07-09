# Release Packet Template

Use this template when assembling evidence for a release decision.

## Required Artifacts
- `Evaluation/runs/<run_id>/cases.jsonl`
- `Evaluation/runs/<run_id>/run_report.json`
- `Evaluation/runs/<run_id>/run_notes.md`
- Validation logs for `/eval-smoke` and `/eval-gate`
- Safety review findings (link to `/safety-review` output)
- Doc updates (`README.md`, `docs/opencode/**`) if behavior changed

## Report Structure
1. **Goal** – what behavior or fix is being released.
2. **Scope** – components touched and any feature flags/config.
3. **Validation** – metrics from `/eval-gate` with measured vs `not_evaluated` domains.
4. **Safety** – summary of `/safety-review` outcomes and remaining risks.
5. **Docs** – links to updated references.
6. **Decision** – go / conditional-go / no-go with justification.

## Blocking Criteria
- Safety or grounding regressions without remediation.
- Missing evaluation artifacts or metadata.
- Contract drift between backend, frontend, and docs.
- Unresolved clinical escalation gaps.

## Conditional-Go Expectations
- Reliability or cost regressions accepted only with explicit owner + follow-up timeline.
- Any temporary mitigations documented in `docs/opencode/memory/` or run notes.
