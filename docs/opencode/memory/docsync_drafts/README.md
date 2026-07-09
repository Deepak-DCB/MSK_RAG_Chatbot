# Doc-Sync Draft Records

This directory stores approved and pending `/doc-sync` draft records.

## Purpose
- Enforce two-step documentation updates: `draft_only` then `apply`.
- Provide auditable references for safe apply operations.

## Required per-draft fields
- `draft_id`
- `created_at`
- `mode` (`draft_only`)
- `goal`
- `target_files`
- `protected_images` (README image refs observed at draft time)
- `proposed_additions`
- `proposed_clarifications`
- `blocked_destructive_requests`
- `safety_checklist`

## File naming
- `<draft_id>.md` (example: `docsync-20260317-001.md`)

## Apply rule
- `mode=apply` must reference an existing `draft_id` and include `approved=true`.
- Apply operations should only implement approved draft scope.
