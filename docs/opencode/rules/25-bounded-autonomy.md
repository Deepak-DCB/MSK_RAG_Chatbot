# Bounded Autonomy Rules

This repository supports controlled optimization loops, not open-ended self-improvement.

## Protected surfaces (human-only unless explicit request)
- Medical safety rules and red-flag escalation criteria/wording.
- Citation-grounding policy and support labeling policy.
- Eval-gate thresholds and release-gating rules.
- Backend safety caps.
- Telemetry contract fields and metadata schema.
- Constitutional safety/grounding/release docs in `docs/opencode/rules/**` and contract docs in `docs/opencode/context/**`.

## Allowed autonomous tuning surfaces
- Retrieval pool sizes.
- Per-source pool sizes.
- Reranker enablement and reranker top-N.
- Biasing parameters.
- Context packing limits.
- Other narrow retrieval/ranking levers that preserve deterministic and inspectable behavior.

## Mandatory loop for autonomous or semi-autonomous tuning
1. Establish baseline.
2. Change one variable only.
3. Run dry-run eval first.
4. Run bounded eval with explicit scope/cost limits.
5. Run grounding/citation support checks.
6. Run safety review.
7. Compare against baseline.
8. Keep only if safety/grounding do not regress.
9. Otherwise revert or recommend revert.
10. Document exactly what changed and why.

## Default stance
- Automate search, not authority.
- Recommend changes freely within allowed surfaces.
- Never auto-merge protected-surface changes.
- When unsure, stop at reviewable recommendations.
