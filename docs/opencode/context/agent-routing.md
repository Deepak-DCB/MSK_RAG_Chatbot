# Agent Routing Matrix

Use the orchestrator to assign work according to this matrix so every task touches the right specialists and safety gates.

## Task-to-Agent
| Task type | Primary agent | Required follow-ups |
| --- | --- | --- |
| Feature or backend contract change | orchestrator → backend-api | evaluation-engineer → safety-auditor → release-manager |
| Retrieval/ranking tuning | orchestrator → retrieval-engineer | evaluation-engineer → safety-auditor → release-manager |
| Frontend UX / telemetry rendering | orchestrator → frontend-integration | evaluation-engineer (if telemetry shape changed) → safety-auditor (if messaging changed) |
| Documentation-only updates | orchestrator → librarian | release-manager (only if docs gate a release) |
| Evaluation pipeline maintenance | orchestrator → evaluation-engineer | safety-auditor (spot-check) → release-manager |
| Clinical reasoning content review | orchestrator → clinical-reasoner | safety-auditor |

## Sequencing Rules
- Always start multi-step work with `/plan-change` so the orchestrator can confirm scope and routing.
- Implementation agents (`backend-api`, `frontend-integration`, `retrieval-engineer`) should not mark tasks done until evaluation and safety subtasks run.
- Docs changes must flow through `librarian` even if implementation agents drafted the wording.
- Release decisions (`go`, `conditional-go`, `no-go`) belong to `release-manager` after evaluation + safety evidence is attached.
