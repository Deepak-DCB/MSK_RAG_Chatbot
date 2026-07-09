# Retrieval and RAG Rules

Prioritize inspectable retrieval behavior over opaque prompt tricks.

## Pipeline expectations
- Preserve the agentic query path (vagueness gate, classify, rewrite, retrieve, rank, pack, answer).
- Keep hybrid dense + BM25 retrieval and deterministic context assembly principles.
- Keep reranker behavior explicit (`off`, `per_source`, future modes must be named).

## Change discipline
- Modify one ranking lever at a time when tuning.
- Measure before and after with production-faithful evaluation.
- Record meaningful retrieval deltas and tradeoffs.

## Reliability
- Do not remove deterministic fallbacks without replacement safeguards.
- Keep retrieval confidence and citations available for inspection.
