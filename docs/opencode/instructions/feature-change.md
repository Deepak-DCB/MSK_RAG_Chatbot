# Instruction Template: Feature Change

Use this template for production feature work.

## Inputs
- Feature goal
- User impact
- Constraints (safety, cost, latency)

## Required output
1. Scope and affected files
2. Implementation plan in ordered steps
3. Safety and regression risks
4. Validation plan (unit, integration, eval)
5. Rollback strategy

## Rules
- Keep changes minimal and auditable.
- Preserve runtime contracts unless explicitly changed.
- Update docs/opencode context when behavior changes.
