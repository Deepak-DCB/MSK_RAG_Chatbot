# Codebase Map

This file is an opencode-facing navigation index. Use it to choose the first files to inspect before deeper code review. It is not a substitute for source inspection when changing behavior.

## Fast Orientation

| Area | Start here | Purpose |
| --- | --- | --- |
| Backend API | `backend/main.py` | FastAPI app, `/health`, `/ask`, `/ask/stream`, `/history`, auth hooks, rate limits, request config allowlist |
| Core RAG | `VectorDB/qaEngine.py` | Query preflight, retrieval, ranking, context packing, prompt construction, OpenAI generation |
| Frontend | `frontend/app.js`, `frontend/index.html`, `frontend/styles.css` | Static guest chat UI, SSE streaming, citations, telemetry, feedback controls |
| Evaluation | `scripts/run_eval_production.py` | Production-faithful eval runner and artifact writer |
| Hierarchical corpus | `scripts/build_hierarchical_corpus.py`, `VectorDB/hierarchical_retrieval.py` | Article/section/span artifacts and citation mapping |
| Concept graph | `scripts/build_concept_graph.py`, `VectorDB/graph_*.py` | Deterministic mechanism graph build and retrieval layer |
| Deployment | `render.yaml`, `frontend/vercel.json` | Render backend and Vercel frontend deployment contracts |
| Project rules | `AGENTS.md`, `docs/opencode/rules/*.md` | Safety, retrieval, evaluation, backend, and autonomy constraints |

## Runtime Flow

1. Browser loads `frontend/index.html`, `frontend/styles.css`, and `frontend/app.js`.
2. `frontend/app.js` sends questions to `POST /ask/stream` and parses SSE token and done events.
3. `backend/main.py` receives the request, enforces server-side caps, builds a `QAConfig`, and calls `VectorDB.qaEngine.agentic_run`.
4. `VectorDB/qaEngine.py` runs local preflight checks, query classification/rewrite, hybrid retrieval, optional reranking, deterministic context packing, prompt construction, and answer generation.
5. `backend/main.py` streams tokens and returns final metadata, citations, telemetry, and completion/error state.
6. `frontend/app.js` renders the answer, citations, evidence spans, mechanism graph details, telemetry, and local-only feedback controls.

## Backend Map

`backend/main.py` is the canonical API surface.

Key anchors:
- `_check_rate_limit` enforces per-IP request limits.
- `_client_ip` resolves proxy-aware client IPs.
- `AskRequest` and `AskResponse` define request/response models.
- `_build_config` clamps and allowlists public request overrides into `QAConfig`.
- `health` implements `GET /health`.
- `ask` implements non-streaming `POST /ask`.
- `ask_stream` implements SSE `POST /ask/stream`.
- `get_history` implements authenticated `GET /history`.
- `_get_supabase`, `_extract_user_id`, and `_save_conversation` support optional auth/history storage.

Backend contract constraints:
- Preserve question length, history turn, output token, and IP rate caps.
- Preserve required telemetry fields consumed by frontend and eval code.
- Public config overrides should remain allowlisted and bounded.
- Failed/incomplete streams must include `error`, `request_id`, and `complete: false` in final metadata.

## Core RAG Map

`VectorDB/qaEngine.py` is the canonical retrieval and answer engine.

Configuration and data structures:
- `QAConfig` controls retrieval mode, pool sizes, token budgets, reranker mode, and model settings.
- `ContextPack` carries packed chunks, graph context, citations, spans, token counts, and telemetry.

Retrieval and ranking anchors:
- `Backend` wraps Chroma collection access.
- `BM25Index` provides lexical retrieval.
- `hybrid_search` fuses dense and BM25 results.
- `merge_raw_results` fuses multiple raw result sets with reciprocal rank fusion.
- `raw_top_confidence` estimates retrieval confidence.
- `generate_multi_queries` expands low-confidence queries.
- `apply_bias` applies deterministic topic/section biasing.
- `maybe_rerank` applies optional per-source reranking.
- `pick_multichunk_context` selects bounded multi-source chunks.
- `compress_context` trims context to relevant sentence-level evidence.
- `build_context_pack` assembles article/section/span/graph context and public citation metadata.

Query and safety anchors:
- `detect_red_flags` detects urgent patterns before normal RAG.
- `_red_flag_response` returns urgent in-person evaluation guidance.
- `detect_scope_issue` catches off-topic, diagnosis-boundary, and medication/treatment-boundary prompts.
- `_scope_boundary_response` returns local scope-boundary guidance.
- `local_preflight` runs deterministic pre-retrieval response gates.
- `_is_vague_query` triggers compact clarification for underspecified prompts.
- `classify_query` and `rewrite_query` prepare retrieval-oriented queries.

Generation anchors:
- `format_context_block` formats retrieved context.
- `build_prompt` builds the grounded answer prompt.
- `ask_openai_llm` calls the generation model.
- `agentic_run` orchestrates the online query path used by the backend.
- `run_qa` is the high-level programmatic QA entry point used by eval and scripts.

Core RAG constraints:
- Keep the deterministic preflight gates before retrieval/generation.
- Preserve hybrid dense + BM25 retrieval unless a task explicitly changes retrieval strategy.
- Change one ranking lever at a time and evaluate before retaining it.
- Do not represent graph paths or retrieved mechanisms as diagnoses.

## Frontend Map

Canonical frontend files:
- `frontend/index.html` defines the static DOM, chat layout, welcome screen, sidebar, auth modal stubs, and script/style links.
- `frontend/styles.css` defines responsive visual layout and UI states.
- `frontend/app.js` handles app state, request streaming, rendering, telemetry panels, citations, and feedback controls.
- `frontend/vercel.json` configures static deployment routing.

`frontend/app.js` key anchors:
- `API_URL` points at the Render backend.
- `REQUEST_CONFIG` carries public reranker overrides only.
- `init` boots health checks and guest mode.
- `setupGuestMode`, `bindAuth`, `setUser`, and `clearUser` manage guest/auth UI state. Current public UI is guest-only.
- `loadSidebarHistory` calls `/history` when authenticated history is available.
- `checkHealth` calls `/health`.
- `sendQuestion` calls `/ask/stream`, parses SSE events, appends streamed tokens, and handles final metadata.
- `createAssistantBubble` and `addMessage` render chat messages.
- `appendCitations`, `appendEvidenceUsed`, `appendMechanismGraph`, and `appendTelemetry` render response details.
- `appendFeedback` renders local-only feedback buttons.
- `renderMarkdown` and `escapeHtml` handle display formatting.

Frontend constraints:
- Keep backend telemetry field usage synchronized with `backend/main.py` and `scripts/run_eval_production.py`.
- Keep public UI guest-only unless a task explicitly changes the product data posture.
- Do not persist free-text health feedback in the guest prototype.
- Maintain mobile and desktop rendering when changing layout.

## Evaluation Map

Canonical runner:
- `scripts/run_eval_production.py`

Key anchors in `scripts/run_eval_production.py`:
- `build_cfg` creates production-like `QAConfig` from CLI args.
- `compute_eval_scope` determines which layers are measured vs `not_evaluated`.
- `make_claims_eval` evaluates citation/grounding checks when labels exist.
- `make_safety_eval` evaluates red-flag escalation behavior when labels exist.
- `make_answer_quality_eval` evaluates topic and uncertainty expectations.
- `make_product_behavior_eval` evaluates zero-cost local behavior gates.
- `make_contexts_used`, `make_hierarchical_eval`, and `make_graph_eval` capture retrieval evidence.
- `mk_case_record` writes per-case artifacts.
- `validate_case_record` and `validate_run_report` enforce artifact shape.
- `summarize_run` builds aggregate metrics.
- `write_run_notes` writes human-readable run notes.
- `main` owns CLI execution.

Eval datasets:
- `datasets/retrieval-goldens.jsonl` for retrieval labels.
- `datasets/citation-tests.jsonl` for citation/grounding checks.
- `datasets/red-flag-cases.jsonl` for urgent escalation behavior.
- `datasets/triage-cases.jsonl` for topic, uncertainty, and triage expectations.
- `datasets/vague-query-cases.jsonl` for adaptive clarification.
- `datasets/off-topic-cases.jsonl` for scope boundaries.
- `datasets/multi-turn-cases.jsonl` for follow-up behavior.
- `datasets/unsupported-claim-cases.jsonl` for diagnosis/treatment/false-reassurance pressure.
- `datasets/graph-coverage-cases.jsonl`, `datasets/graph-mechanism-cases.jsonl`, and `datasets/graph-completeness-cases.jsonl` for concept graph coverage.

Eval artifact contract:
- `Evaluation/runs/<run_id>/cases.jsonl`
- `Evaluation/runs/<run_id>/run_report.json`
- `Evaluation/runs/<run_id>/run_notes.md`

Eval constraints:
- Run dry-run before paid/bounded evaluation.
- Keep non-measured layers marked as `not_evaluated`.
- Include commit hash, dataset hash/version, pipeline mode, and key config values in run metadata.

## Hierarchical Corpus Map

Build script:
- `scripts/build_hierarchical_corpus.py`

Runtime helpers:
- `VectorDB/hierarchical_retrieval.py`

Key anchors:
- `build_hierarchical_corpus` builds article, section, chunk, and evidence-span artifacts from source chunks.
- `HierarchicalCorpus` stores article/section/span indexes.
- `load_hierarchical_corpus` loads committed hierarchical artifacts.
- `map_chunks_to_hierarchy` maps selected chunks into article/section/span context.
- `build_citation_map` builds citation metadata for frontend/eval visibility.
- `reconstruct_article_context` and `reconstruct_section_context` rebuild larger context windows when needed.

Expected artifact family:
- `MSKArticlesINDEX/hierarchical/` contains generated article/section/span artifacts when present.
- `MSKArticlesINDEX/` also contains source-index utilities and legacy preprocessing scripts.

## Concept Graph Map

Build script:
- `scripts/build_concept_graph.py`

Runtime helpers:
- `VectorDB/graph_vocab.py` defines canonical concepts and aliases.
- `VectorDB/graph_paths.py` builds deterministic mechanism paths from nodes and edges.
- `VectorDB/graph_retrieval.py` retrieves graph nodes, edges, paths, and supporting spans for query context.

Key anchors:
- `graph_vocab.detect_entities` maps text to canonical concepts.
- `build_concept_graph` emits graph nodes, edges, claims, paths, and manifest metadata.
- `build_mechanism_paths` creates conservative graph path records.
- `graph_retrieval.load_graph` loads graph artifacts.
- `graph_retrieval.build_graph_context` prepares graph context for `qaEngine.build_context_pack`.
- `graph_retrieval.format_graph_context` formats graph context for prompts.

Graph constraints:
- Current graph is broader than the early TOS-biased baseline but still not corpus-complete or clinically validated.
- Do not claim full 20-article completeness unless coverage, false-positive aliases, and clinician-review constraints are resolved.
- Preserve conservative policy labels for weak, indirect, inferred, or safety-critical paths.
- Guard known alias collisions and overbroad anatomical edges.

## Tests Map

Core test files:
- `tests/test_backend_contracts.py` checks API contract behavior.
- `tests/test_safety_gate.py` checks deterministic red-flag and boundary behavior.
- `tests/test_eval_harness.py` checks eval runner behavior and artifact shape.
- `tests/test_frontend_guest_ui.py` checks guest UI/product posture expectations.
- `tests/test_hierarchical_corpus.py` checks hierarchical corpus helpers.
- `tests/test_concept_graph_build.py` checks graph build correctness guards.
- `tests/test_graph_retrieval.py` checks graph retrieval formatting/context behavior.
- `tests/test_graph_coverage.py` and `tests/test_graph_completeness_dataset.py` check concept graph datasets and coverage expectations.

CI:
- `.github/workflows/ci.yml` compiles key modules, runs `pytest tests -q`, and runs `python scripts/run_eval_production.py --dry-run --max-cases 2`.

## Data And Generated Artifacts

Committed runtime data:
- `chroma_store/` contains the Chroma vector collection used by the backend.
- `embeddings/embedding_model.txt` records the embedding model used for source embeddings.

Large or generated data families:
- `Embedding/*.csv`, `Embedding/*.npy`, and related scripts are embedding-generation artifacts and utilities.
- `MSKArticlesINDEX/` contains source extraction/index artifacts, hierarchical corpus outputs, graph outputs, and older utilities.
- `Evaluation/runs/` is the expected location for production eval outputs.

Secrets and sensitive files:
- `.env` and `.env.*` must never be committed or copied into docs.
- Supabase service keys, JWT secrets, and OpenAI keys are server-only secrets.

## Legacy Or Prototype Paths

Do not treat these as production defaults unless the user explicitly asks:
- `app.py` legacy Streamlit/prototype app.
- `runAll.py` legacy orchestration script.
- `VectorDB/retrieval.py` older local retrieval CLI.
- `VectorDB/ChromaDB.py`, `scripts/rebuild_chroma_openai.py`, `Embedding/*.py`, and `Text_Extraction/*.py` are build/preprocessing utilities, not online runtime endpoints.
- `chatbot/mskbot.py`, `chunk_editor.py`, `reviewGoldset.py`, and notebook files are tooling/prototype surfaces.

## Change Routing Table

| If the task is about... | Inspect first | Validate with |
| --- | --- | --- |
| API endpoint behavior | `backend/main.py`, `docs/opencode/context/runtime-contracts.md` | `pytest tests/test_backend_contracts.py -q` |
| Streaming/SSE UI behavior | `backend/main.py`, `frontend/app.js` | backend contract tests plus manual stream check if feasible |
| Retrieval ranking or context quality | `VectorDB/qaEngine.py`, `VectorDB/hierarchical_retrieval.py`, `VectorDB/graph_retrieval.py` | eval dry-run, retrieval datasets, grounding/safety review |
| Red flags or scope boundaries | `VectorDB/qaEngine.py`, `tests/test_safety_gate.py`, `docs/opencode/rules/15-medical-safety.md` | `pytest tests/test_safety_gate.py -q` and safety review |
| Frontend telemetry rendering | `frontend/app.js`, `backend/main.py`, `scripts/run_eval_production.py` | `pytest tests/test_frontend_guest_ui.py -q` and contract check |
| Eval runner or metrics | `scripts/run_eval_production.py`, `docs/opencode/context/evaluation-contracts.md` | `python scripts/run_eval_production.py --dry-run --max-cases 2` |
| Concept graph coverage | `VectorDB/graph_vocab.py`, `VectorDB/graph_paths.py`, `scripts/build_concept_graph.py` | graph tests and graph coverage datasets |
| Deployment config | `render.yaml`, `frontend/vercel.json`, `backend/requirements.txt` | config review and smoke tests |
| Documentation or opencode context | `AGENTS.md`, `docs/opencode/**`, `README.md` | no runtime tests unless behavior claims changed |

## Protected Surfaces

Treat these as human-sensitive or release-gated unless the user explicitly asks for changes:
- Medical safety rules and red-flag wording.
- Citation-grounding policy and support labeling policy.
- Eval-gate thresholds and release-gating rules.
- Backend safety caps and request config allowlist.
- Telemetry contract fields and metadata schema.
- Constitutional project rules under `docs/opencode/rules/**`.

## Opencode Context Integration

`opencode.json` includes this glob in `instructions`:

```json
"docs/opencode/context/*.md"
```

Because this file lives at `docs/opencode/context/codebase-map.md`, it is loaded automatically as project context by opencode. If `opencode.json` changes in the future, preserve this file or an equivalent codebase-map path in the `instructions` list.
