# Modern RAG Upgrade

This upgrade adds hierarchical, long-context support without replacing the current FastAPI app, Chroma store, safety gates, SSE streaming, or evaluation runner.

## What Changed

- Added first-class reconstructed corpus artifacts under `MSKArticlesINDEX/hierarchical/`.
- Added `VectorDB/hierarchical_retrieval.py` to load articles, sections, and evidence spans.
- Added `context_strategy` support in `VectorDB/qaEngine.py`:
  - `chunk_pack` keeps the previous chunk-only behavior.
  - `section_expand` expands retrieved chunks into reconstructed sections.
  - `article_expand` expands retrieved chunks into reconstructed article text when budget allows.
  - `hybrid_long_context` uses selected reconstructed sections plus compact evidence spans and the original selected chunks.
- Fixed query rewrite handling so the rewritten query is used for retrieval while generation answers the original user question by default.
- Added optional response metadata for evidence spans, citation maps, selected articles, selected sections, context strategy, and fallback reason.

## Hierarchical Artifacts

Build artifacts with:

```bash
python scripts/build_hierarchical_corpus.py
```

Outputs:

- `MSKArticlesINDEX/hierarchical/articles.jsonl`
- `MSKArticlesINDEX/hierarchical/sections.jsonl`
- `MSKArticlesINDEX/hierarchical/paragraphs.jsonl`
- `MSKArticlesINDEX/hierarchical/evidence_spans.jsonl`
- `MSKArticlesINDEX/hierarchical/corpus_manifest.json`

The current implementation reconstructs from `MSKArticlesINDEX/chunks.parquet` because the original HTML mirror is not required for this phase. Article and section text is rebuilt by grouping `source_relpath`, sorting by `article_seq`, and suppressing obvious exact overlap between adjacent chunks.

## Context Strategy

The backend default is `hybrid_long_context` through `QAConfig.context_strategy`.

If hierarchical artifacts are unavailable, the runtime falls back to `chunk_pack` and returns:

- `context_strategy: "chunk_pack"`
- `fallback_reason: "hierarchical_artifacts_missing"` or another concrete reason
- `hierarchical_available: false`

The public backend config remains intentionally narrow. Frontend requests still only override reranker settings, so context strategy remains server-owned by default.

## Evidence Metadata

Existing `citations` remain source/section strings for frontend compatibility.

New optional metadata includes:

- `evidence_spans`
- `citation_map`
- `selected_articles`
- `selected_sections`
- `context_strategy`
- `fallback_reason`
- `hierarchical_available`
- `context_token_estimate`
- `original_question`
- `refined_query`

The frontend adds a compact expandable `Evidence used` section when evidence spans are present.

## Running The App

Backend:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 10000
```

Frontend remains the static app in `frontend/` and still calls the deployed API URL configured in `frontend/app.js`.

## Running Evaluation

Dry-run validation:

```bash
python scripts/run_eval_production.py --dry-run --max-cases 5
```

Production-faithful bounded run:

```bash
python scripts/run_eval_production.py --max-cases 10 --price-input-per-1k 0.0 --price-output-per-1k 0.0
```

The eval artifacts now record hierarchical context metadata and summary metrics for evidence-span presence, citation-support overlap proxy, original-question preservation, strategy usage, fallback rate, and artifact availability.

## Known Limitations

- Full article text is reconstructed from chunks, not raw HTML.
- Paragraph records are chunk-derived and do not guarantee original paragraph boundaries.
- Evidence spans are sentence groups, not clinician-validated support links.
- There is no new vector index for article, section, or evidence-span retrieval yet.
- The existing reranker remains disabled by default because prior checked-in evidence showed worse retrieval results.
- No external medical/anatomy sources were added in this phase.
- Claim-to-evidence verification remains a rule-based proxy, not a semantic verifier or clinician review.

## Future Work

- Add raw HTML re-ingestion mode when original article mirrors are available.
- Add multi-granularity vector and lexical indexes for article, section, paragraph, and evidence span retrieval.
- Add a measured modern reranker only if bounded evaluation shows improvement.
- Add semantic claim extraction and claim-to-span grounding checks.
- Add an evaluation dashboard over `Evaluation/runs/<run_id>/` artifacts.
- Add provider-level prompt caching boundaries if the selected LLM supports them.
