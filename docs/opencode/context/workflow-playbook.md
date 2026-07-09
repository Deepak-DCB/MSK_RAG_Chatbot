# Workflow Playbook

Use command-driven flows to keep work goal-oriented and auditable.

## Feature / Backend Change
1. `/plan-change <summary>` – orchestrator scopes goal, files, risks, validation.
2. `/implement-change <plan>` – build agent makes minimal edits; report touched files + validation.
3. `/eval-smoke` – evaluation-engineer runs dry-run to confirm artifacts.
4. `/safety-review` – safety-auditor checks for clinical/grounding regressions.
5. `/eval-gate` – evaluation-engineer reports bounded metrics + blockers.
6. `/doc-sync` – librarian updates README + `docs/opencode/**` if behavior changed.
7. `/plan-change` (optional) – orchestrator summarizes final decision before release.

## Retrieval / Ranking Tuning
1. `/plan-change` – capture retrieval lever(s) being tuned and baseline metrics.
2. Implement change via `retrieval-engineer` (still using `/implement-change`).
3. `/rag-audit` – evaluation-engineer inspects retrieval recall, ranking, citations.
4. `/eval-gate` – bounded run with measured vs `not_evaluated` metrics.
5. `/safety-review` – ensure new retrieval behavior does not weaken escalation guidance.
6. If safety/grounding regress or metrics degrade unacceptably, revert or recommend revert.
7. `/doc-sync` – record tuned lever, baseline/delta, and decision rationale in `docs/opencode/`.

## Documentation-Only Refresh
1. `/doc-sync <goal> mode=draft_only` – librarian collects context and creates draft record.
2. Review draft (`draft_id`) with protected image inventory and safety checklist.
3. `/doc-sync mode=apply draft_id=<id> approved=true` – apply approved additive/clarifying edits only.
4. `/plan-change` (optional) – orchestrator confirms no runtime edits required.
5. `/release-packet` (optional) – capture if docs gate a release or rollback note.

## README Protection Checklist
- Preserve existing README image tags and URLs.
- Do not remove README sections unless explicit destructive approval is given.
- Prefer additive clarifications and `Needs confirmation` notes for uncertain content.
- Require `draft_only` before any `apply` operation.

## Release Preparation
1. Ensure latest `/implement-change` outputs include validation evidence.
2. `/eval-smoke` (if not run in last 24h) + `/eval-gate` with bounded paid run.
3. `/safety-review` – confirm no clinical regressions.
4. `/doc-sync` – update run notes, README, or deployment docs.
5. `release-manager` (via `/plan-change` or direct prompt) delivers go / conditional-go / no-go referencing `docs/opencode/context/release-packet.md`.
