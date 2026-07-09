# Product Safety Policy v1

This project is moving toward a public educational MSK triage product, but the current productized scope remains conservative.

## Product Boundary
- Educational triage support only.
- No diagnosis, no definitive prognosis, and no replacement for in-person care.
- No emergency-deferring reassurance.
- No treatment prescription beyond low-risk educational guidance grounded in retrieved context.
- Public UI is guest-only for now so the core product can mature without storing user health conversations.
- Medication/dose/injection advice and diagnosis-certainty requests are handled by local scope-boundary responses.

## Guest-Only Data Posture
- Frontend chat history is session-only browser state.
- The visible product does not require accounts or saved history.
- Users should avoid entering identifying information.
- Backend auth/history endpoints may remain available for future use, but they are not part of the current guest-only frontend flow.

## Deterministic Red-Flag Gate
Before retrieval or generation, the runtime checks for high-risk symptom patterns. If triggered, it returns a short urgent in-person evaluation response and does not provide speculative biomechanics.

Urgent gate reasons include:
- bowel or bladder changes
- new or worsening neurologic weakness/numbness
- significant trauma with concerning symptoms
- severe chest pain or breathing symptoms
- fever with systemic decline or severe symptoms
- unexplained weight loss with concerning symptoms

## Adaptive Clarification
Open chat remains fluid. Vague prompts should receive a compact clarification request asking for location, sensation, and trigger/timing, plus a reminder to mention urgent signs.

## Trust UX
- Citations should be expandable enough to show source and section context.
- Technical details belong behind `Why this answer?`, not in the main answer body.
- Feedback controls must not persist free-text health information in the guest-only prototype.

## Evaluation Expectations
- Safety-gate behavior must be covered by local unit tests.
- Red-flag datasets should include urgent cases and non-urgent controls.
- Zero-cost `product_behavior` dry-runs should cover red flags, vague prompts, off-topic prompts, multi-turn follow-ups, and unsupported-claim pressure cases.
- Clinician review is still not available and must not be represented as completed.
