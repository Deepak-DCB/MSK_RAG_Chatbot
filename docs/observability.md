# Observability and Traceability

This project treats the answer as the end of the pipeline, not the only thing worth inspecting.

## What the live system exposes

The FastAPI backend returns structured metadata on both `/ask` and `/ask/stream` final events:

- retrieval confidence
- retrieval and generation timing
- prompt, output, context, and question token counts
- category and category label
- refined query
- triage level
- safety-gate trigger status and reasons
- scope-boundary issue when the local preflight blocks diagnosis, medication, or non-MSK prompts
- reranker mode and `reranker_top_n`
- config source (`default` vs request override)
- citations
- streamed completion metadata (`complete`, `error`, `request_id` on failure)

The frontend renders expandable source details, a `Why this answer?` panel, triage/safety/scope metadata, and local-only feedback buttons. The production eval runner writes the same metadata families into `cases.jsonl`.

## Example `/ask` response shape

This example mirrors the fields defined in `backend/main.py`.

```json
{
  "answer": "...",
  "citations": [
    "mskneurology.com/how-truly-treat-thoracic-outlet-syndrome/index.html"
  ],
  "retrieval_confidence": 0.61,
  "retrieval_time": 0.42,
  "generation_time": 1.17,
  "prompt_tokens": 1788,
  "output_tokens": 356,
  "context_tokens": 1420,
  "question_tokens": 18,
  "category": "structured_biomechanical_pattern",
  "category_label": "Structured biomechanical pattern",
  "refined_query": "thoracic outlet symptoms with scapular depression and arm tingling",
  "triage_level": "educational_triage",
  "safety_gate_triggered": false,
  "safety_gate_reasons": [],
  "scope_issue": null,
  "reranker_mode": "off",
  "use_reranker": false,
  "reranker_top_n": 10,
  "openai_model": "gpt-4.1-mini",
  "config_source": "default"
}
```

## Example `/ask/stream` done event

The SSE endpoint emits token events during generation and a final `done` event with metadata:

```text
event: done
data: {
  "citations": ["mskneurology.com/..."],
  "retrieval_confidence": 0.61,
  "retrieval_time": 0.42,
  "generation_time": 1.17,
  "prompt_tokens": 1788,
  "output_tokens": 356,
  "context_tokens": 1420,
  "question_tokens": 18,
  "category": "structured_biomechanical_pattern",
  "category_label": "Structured biomechanical pattern",
  "refined_query": "thoracic outlet symptoms with scapular depression and arm tingling",
  "triage_level": "educational_triage",
  "safety_gate_triggered": false,
  "safety_gate_reasons": [],
  "scope_issue": null,
  "reranker_mode": "off",
  "use_reranker": false,
  "reranker_top_n": 10,
  "openai_model": "gpt-4.1-mini",
  "config_source": "default",
  "complete": true
}
```

If streaming fails or times out, the same final event includes:

- `complete: false`
- `error`
- `request_id`

That makes failed conversations debuggable without guessing whether the model, transport, or backend failed.

## Why this matters

- A recruiter can see that retrieval is inspectable, not hidden behind a single chat bubble.
- A reviewer can separate ranking failures from generation failures.
- Evaluation artifacts and live telemetry speak the same language: confidence, citations, timings, token usage, and reranker mode.
