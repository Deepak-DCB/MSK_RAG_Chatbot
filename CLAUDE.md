# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSK Triage Chatbot — a domain-constrained RAG system for musculoskeletal biomechanics triage and Q&A. The core artifact is the retrieval pipeline in `VectorDB/qaEngine.py`, not the chat UI. Treat it as a safety-sensitive, evidence-grounded system. The primary goal is inspectable, measurable retrieval quality — not LLM fluency.

**Deployed:** FastAPI on Render (`https://msk-rag-chatbot.onrender.com`), static frontend on Vercel (`https://msk-triage-chatbot.vercel.app`).

## Development Commands

### Run the backend locally
```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Run the static frontend locally
```bash
python -m http.server 5500 --directory frontend
```
The frontend hardcodes the Render URL at `frontend/app.js:3` (`API_URL`). Change it to point at a local backend for local development.

### Run tests
```bash
pip install -r backend/requirements.txt pytest httpx
pytest tests -q
```

### Run a single test file
```bash
pytest tests/test_backend.py -q
```

### Eval dry-run (no API calls — always run this first)
```bash
python scripts/run_eval_production.py --dry-run --max-cases 5
```

### Bounded paid eval (cost-guarded)
```bash
python scripts/run_eval_production.py --max-cases 10 --price-input-per-1k 0.001 --price-output-per-1k 0.004 --max-estimated-cost-usd 1.00
```

### Dataset-specific eval commands
```bash
# Citation/grounding checks (paid)
python scripts/run_eval_production.py --dataset datasets/citation-tests.jsonl --max-cases 3

# Red-flag safety-gate checks (zero-cost, dry-run)
python scripts/run_eval_production.py --dataset datasets/red-flag-cases.jsonl --dry-run --max-cases 50

# Vague-query clarification checks (zero-cost)
python scripts/run_eval_production.py --dataset datasets/vague-query-cases.jsonl --dry-run --max-cases 25

# Off-topic/scope-boundary checks (zero-cost)
python scripts/run_eval_production.py --dataset datasets/off-topic-cases.jsonl --dry-run --max-cases 25

# Multi-turn behavior checks (zero-cost)
python scripts/run_eval_production.py --dataset datasets/multi-turn-cases.jsonl --dry-run --max-cases 10

# Unsupported-claim pressure checks (zero-cost)
python scripts/run_eval_production.py --dataset datasets/unsupported-claim-cases.jsonl --dry-run --max-cases 10

# Triage answer-quality checks (paid)
python scripts/run_eval_production.py --dataset datasets/triage-cases.jsonl --max-cases 3
```

### Summarize checked-in ablation results
```bash
python scripts/summarize_eval_results.py --format markdown
```

### Compile-check key modules (syntax only)
```bash
python -m py_compile backend/main.py VectorDB/qaEngine.py scripts/run_eval_production.py
```

### CLI mode for qaEngine (interactive REPL)
```bash
python VectorDB/qaEngine.py --q "your question here"
```

## Required Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Embeddings (`text-embedding-3-large`) and generation (`gpt-4.1-mini`, `gpt-4.1-nano`) |
| `SUPABASE_URL` | No | Auth / history persistence |
| `SUPABASE_SERVICE_KEY` | No | Supabase service role |
| `SUPABASE_JWT_SECRET` | No | JWT decode fallback |
| `CORS_ORIGINS` | No | Comma-separated extra CORS origins |
| `TRUST_PROXY_HEADERS` | No | Set to `1` to enable X-Forwarded-For |

Production (Render) only configures `OPENAI_API_KEY` and `CORS_ORIGINS`.

## Architecture

```
Browser (Vercel static)
  └── frontend/app.js  ──POST /ask/stream──►  backend/main.py  (Render)
                                                     │
                                         sys.path insert PROJECT_ROOT/VectorDB
                                                     │
                                         VectorDB/qaEngine.py  agentic_run()
                                              │          │            │
                                         ChromaDB    OpenAI API  MSKArticlesINDEX/
                                         chroma_store/ (embed+gen) hierarchical/
                                                                   graph/
                                                                   mechanics/
```

### Request pipeline in `agentic_run()` (`VectorDB/qaEngine.py`)

1. **`local_preflight()`** — zero-cost deterministic gates: red-flag regex → scope boundary regex → vagueness check. Short-circuits before any API call.
2. **`classify_query()`** — `gpt-4.1-nano` classifies query type.
3. **`rewrite_query()`** — `gpt-4.1-nano` rewrites into MSK-optimized form using conversation history. The rewritten query is used for retrieval only; generation receives the original user question (`answer_original_question=True`).
4. **`hybrid_search()`** — ChromaDB dense (OpenAI embeddings) + BM25 (`rank_bm25`) fused with Reciprocal Rank Fusion.
5. **Optional multi-query expansion** — only when initial confidence < 0.33.
6. **`apply_bias()`** — additive distance adjustments using `GOOD_SECTIONS`/`NARRATIVE_SECTIONS` exact sets and pattern fallbacks.
7. **`maybe_rerank()`** — LLM scoring via `gpt-4.1-nano`. **Disabled by default** — ablation shows Hit@5 drops from 94% to 38%.
8. **`pick_multichunk_context()`** — packs chunks into 10,000-token budget, max 3 chunks per source.
9. **`compress_context()`** — keyword-overlap sentence filtering, keeps ~65%.
10. **`build_context_pack()`** — assembles hierarchical section/article context + concept graph paths + evidence spans. Default strategy: `hybrid_long_context`, falls back to `chunk_pack` when artifacts missing.
11. **`build_prompt()` + `ask_openai_llm()`** — streams answer via `gpt-4.1-mini`.

**Streaming:** `/ask/stream` runs `agentic_run()` on a daemon thread with an `on_token` callback, yields SSE `data: {"token": "..."}` events, ends with `event: done` + full telemetry metadata. 120s timeout.

### Document hierarchy

The corpus is organized as:
```
Article → Section → Paragraph → Evidence span
```
Hierarchical artifacts live in `MSKArticlesINDEX/hierarchical/`. The concept graph artifacts live in `MSKArticlesINDEX/graph/` (nodes, edges, paths, claims, manifest). Graph context has a compact default budget of `graph_max_tokens=1800` and is designed to focus context, not enlarge prompts.

## Key Design Decisions

**`VectorDB/` is not a Python package.** No `__init__.py` exists. Both `backend/main.py` and eval scripts insert `PROJECT_ROOT/VectorDB` into `sys.path` at startup. All sub-modules use try/except for graceful fallback.

**Two requirements files.** `backend/requirements.txt` is the minimal production install (used by Render and CI). Root `requirements.txt` adds dev/script extras (torch, sentence-transformers, pandas, bs4). Never use the root file in production.

**`chroma_store/` is committed.** The ChromaDB collection is a checked-in runtime artifact. Rebuilding it requires `scripts/rebuild_chroma_openai.py` with a live OpenAI key. All `MSKArticlesINDEX/` artifacts are similarly committed outputs — rebuild through scripts, never hand-edit.

**Reranker is permanently OFF by default.** `use_reranker: False` in `QAConfig` and `use_reranker: false` in `frontend/app.js` `REQUEST_CONFIG`. Public request config only allows `use_reranker` and `reranker_top_n` as user-tunable — all other config keys are silently ignored.

**Auth UI exists but is disabled.** `AUTH_ENABLED = false` in `app.js`. The HTML auth modal is inert; only guest mode is active.

**History injection uses direct message injection, not vector memory.** Multi-turn context goes into the LLM messages array via `_truncate_history()`. The vector-memory history path (`include_history=False` in `QAConfig`) is off by default.

**Context strategy default is `hybrid_long_context`.** Expands to section-level text + graph context + evidence spans + chunk context, subject to per-level token caps (`max_section_context_tokens=2500`, `max_article_context_tokens=6000`). Automatic fallback to `chunk_pack` when hierarchical artifacts are missing.

**The concept graph is evidence-grounded, not diagnostic.** It represents possible mechanism chains (e.g., `scapular depression → clavicle → costoclavicular space → brachial plexus → neuralgia`) but cannot make causal claims. Edges carry `support_level`, `claim_strength`, and `clinical_risk` labels. When `graph_focus_context=True` and useful paths are found, broad section/article expansion is avoided in favour of compact graph context.

## Artifact Rebuild Pipeline

Run in order when rebuilding from scratch:
```bash
python Text_Extraction/textExtract.py            # HTML → MSKArticlesINDEX/chunks.parquet
python scripts/rebuild_chroma_openai.py          # chunks.parquet → chroma_store/
python scripts/build_hierarchical_corpus.py      # chunks.parquet → MSKArticlesINDEX/hierarchical/
python scripts/build_concept_graph.py            # hierarchical/ → MSKArticlesINDEX/graph/
python scripts/build_mechanics_maps.py           # → MSKArticlesINDEX/mechanics/
```

## Evaluation

**Checked-in ablation evidence** (50 gold-set questions):

| Variant | Hit@1 article | Hit@5 chunk | NDCG@5 |
|---|---|---|---|
| Topic-aware baseline | 98.0% | 94.0% | 0.787 |
| Topic-aware + reranker | 60.0% | 38.0% | 0.273 |

The negative reranker result is intentionally kept visible.

**Artifact locations:**
- Gold set inputs: `Eval/gold_set_v2.jsonl`
- Eval run outputs: `Evaluation/runs/<run_id>/cases.jsonl`, `run_report.json`, `run_notes.md`
- Schema refs: `Evaluation/automation_eval_schema_v2.json`, `Evaluation/clinician_review_rubric.txt`

**Evaluation layers** — each is reported separately:
- `retrieval` — Hit@N, MRR, NDCG. Strongest current evidence.
- `grounding` — required source citation rate, rule-based claim-support match. Still rule-based proxies.
- `safety` — escalation recall/precision, false reassurance rate. Dry-runs mark this `not_evaluated`.
- `product_behavior` — zero-cost local checks for safety-gate trigger, clarification, scope boundary, diagnosis-boundary. Does not measure semantic answer quality.
- `answer_quality` — topic coverage, required uncertainty pass rate. Requires paid model calls.

Non-measured layers are marked `not_evaluated`, never zero. The eval runner auto-detects dataset types to activate the corresponding checks.

## Red-Flag Escalation

`local_preflight()` runs deterministic red-flag checks before any retrieval or generation. Escalate to urgent in-person evaluation when prompts suggest:
- New or worsening neurologic weakness or numbness
- Bowel or bladder changes
- Significant trauma with concerning symptoms
- Severe chest pain or breathing symptoms
- Fever with systemic decline
- Unexplained weight loss with persistent pain
- Rapidly progressive or unusual symptoms

Never add reassurance language that might deflect urgent care. These gates are protected surfaces.

## Telemetry Response Shape

Both `/ask` and the SSE `done` event include:
```json
{
  "answer": "...",
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
  "refined_query": "...",
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
On streaming failure: `complete: false`, `error`, `request_id`. Any change to this shape must be coordinated with frontend and eval updates simultaneously.

## Change Discipline

- **Change one retrieval/ranking lever at a time.** Always establish baseline metrics before changing anything.
- **Dry-run eval before any paid run.** Always.
- **Protected surfaces** — do not change without an explicit request: medical safety rules, red-flag escalation wording, backend safety caps (question length cap 1000, history turn cap 5, output token cap 1000, per-IP rate limit 5/60s), telemetry schema, citation-grounding policy, eval-gate thresholds.
- **Do not enable reranking by default** unless new eval evidence supports it.
- **If safety, grounding, or retrieval regressions appear** — revert or recommend revert. Do not retain them.
- **Retrieval/ranking changes include:** dense pool sizes, BM25 behavior, hybrid fusion, query rewriting, reranker mode/top-N, source balancing, context packing, hierarchical expansion, graph context inclusion, biasing parameters.

## CI

`.github/workflows/ci.yml`: Python 3.11, installs backend deps + pytest, compile-checks four key modules, runs `pytest tests -q`, runs eval with `--dry-run --max-cases 2`.

## What Not to Touch

- `app.py`, `runAll.py`, `chatbot/mskbot.py` — legacy Streamlit artifacts, not production.
- Root-level diagnostic JSONL files (`diag_gpt-4.1_*.jsonl`) — historical eval outputs.
- `opencode.json` / `.opencode/` / `docs/opencode/` — config and rules for the OpenCode AI editor, not Claude Code. Read them for context, but do not edit them.
- `MSKArticlesINDEX/` and `chroma_store/` — generated runtime artifacts. Rebuild through scripts; never hand-edit.
