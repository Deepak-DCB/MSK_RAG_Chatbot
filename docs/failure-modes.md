# Failure Modes and Mitigations

This repository is more useful as a portfolio project if it makes failures explicit.

## 1. Retrieval finds the wrong article

Why it happens:

- symptom wording does not match the corpus vocabulary
- semantically nearby but clinically different topics cluster together
- short follow-ups lose anatomical context

Current mitigations:

- history-aware query classification and rewriting in `VectorDB/qaEngine.py`
- hybrid dense + BM25 retrieval fused with reciprocal rank fusion
- adaptive multi-query expansion only when confidence is weak
- topic and section biasing that is explicit and inspectable

## 2. Retrieval finds the right article but weak chunks

Why it happens:

- long-form clinical articles mix mechanism sections with narrative sections
- dense similarity alone does not respect section quality

Current mitigations:

- section-level boosts for mechanism-dense sections
- penalties for narrative / low-yield sections
- per-source pooling and deterministic context packing

## 3. Reranker makes things worse

Observed evidence:

- the checked-in ablation files show the current per-source reranker underperforming the topic-aware baseline
- `eval_results_topicaware.json` beats `eval_results_topicaware_reranked.json` on article ranking, chunk ranking, and `NDCG@5`

Current mitigation:

- backend public config defaults `use_reranker` to `false`
- README and docs now surface this as a negative result instead of marketing it as a win

## 4. Answer sounds plausible but grounding is thin

Why it happens:

- citation presence is easier to automate than semantic support checking
- a fluent answer can still overstate what the retrieved text supports

Current mitigations:

- citations are returned to the client and written into eval artifacts
- `scripts/run_eval_production.py` now runs explicit grounding checks for `datasets/citation-tests.jsonl`
- unsupported layers are marked `not_evaluated` instead of silently reported as success

Current limitation:

- grounding checks are still rule-based proxies, not full claim extraction plus clinician adjudication

## 5. Safety-critical prompts receive weak escalation

Why it happens:

- general LLM behavior often drifts toward reassurance
- red-flag detection can be missed when symptoms are wrapped in conversational language

Current mitigations:

- deterministic red-flag gate runs before retrieval or generation for high-risk symptom patterns
- conservative safety language in the system prompt
- server-side rate limits and bounded history
- explicit red-flag datasets in `datasets/red-flag-cases.jsonl` and `datasets/triage-cases.jsonl`
- production runner now reports escalation recall, precision, and false reassurance rate when those datasets are used

Current limitation:

- clinician review is not available yet, so safety claims remain automated and conservative rather than clinically adjudicated

## 6. Runtime issues are hard to debug

Why it happens:

- streaming systems can fail after the model starts responding
- retrieval, generation, and transport failures look similar in a basic chat UI

Current mitigations:

- `/ask/stream` emits a final `done` event with `complete`, `error`, and `request_id`
- live responses expose retrieval confidence, timings, token counts, refined query, and reranker mode
- eval artifacts capture the same metadata for offline analysis

## 7. Local gates pass but answer quality is still unmeasured

Why it happens:

- zero-cost dry-runs can evaluate deterministic safety-gate triggers, clarification, and scope-boundary behavior
- normal RAG answer correctness still requires model calls and stronger grounding checks
- clinician review is not available yet

Current mitigations:

- `product_behavior` metrics are separated from retrieval, grounding, safety, and answer-quality metrics
- dry-run artifacts keep non-measured layers marked `not_evaluated`
- case outputs include `response_source` so empty dry-run answers are distinguishable from local preflight responses
- datasets now include red-flag, vague-query, off-topic, multi-turn, and unsupported-claim pressure cases
