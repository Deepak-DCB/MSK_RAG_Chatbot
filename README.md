# MSK RAG Chatbot  
### Biomechanics Clinical Question Answering using MSK Neurology (Retrieval-Augmented Generation)

## TL;DR

**This is:** A retrieval systems engineering case study in a constrained clinical domain.  
**This is not:** A diagnostic tool, autonomous clinician, or end-to-end learned medical model.

- Domain-constrained RAG system for musculoskeletal neurology and biomechanics  
- Retrieval-first design with hybrid dense + sparse search, multi-query expansion, and deterministic context assembly  
- Agentic query classification, rewriting, and conversational follow-up handling  
- No fine-tuning, no end-to-end black box; emphasis on inspectability and failure analysis



This repository implements a **retrieval-augmented question answering (RAG) system** for answering **mechanism-level clinical questions** grounded in a corpus of 20 articles (1,301 chunks) derived from **MSKNeurology.com** (Kjetil Larsen).

Rather than treating the LLM as an end-to-end reasoning engine, the system treats **answer quality as an effect of retrieval quality**, and so is designed to expose, constrain, and debug each step of the retrieval and context-selection process.

The system surfaces retrieved chunks, distances, heuristic adjustments, reranking behavior, token budgets, latency, and confidence signals so that outputs can be **inspected, audited, and failure-mode analyzed**.

> **Note:** This system is not a medical device and does not provide diagnoses or treatment recommendations. It is an educational and research-oriented explainer grounded strictly in retrieved corpus content.

---

## Motivation

Musculoskeletal neurology is a narrow domain where valid explanations depend on **anatomy, biomechanics, and neurovascular space**, typically described in long-form clinical articles rather than structured knowledge bases.

General-purpose language models frequently hallucinate or over-generalize in this domain when used without strong retrieval constraints.

This project explores how far **explicit retrieval design, deterministic context assembly, and domain-encoded heuristics**—rather than increasingly complex prompting or fine-tuning—can improve answer traceability, interpretability, and robustness in a specialized clinical corpus.

---

## System overview

The system is structured as a **three-stage pipeline**: offline corpus processing, persistent retrieval infrastructure, and online query-time reasoning.

### 1. Offline corpus processing

- HTML articles from MSKNeurology.com are mirrored locally.
- Text is cleaned and segmented using **sentence-first, token-aware chunking**.
- Each chunk is annotated with article, section, position, and token-length metadata.
- Outputs are persisted as a structured chunk table (`chunks.parquet`) used for downstream retrieval.

### 2. Persistent retrieval infrastructure

- Dense embeddings are generated for all chunks using **OpenAI `text-embedding-3-large`** (3072-dim), called via the embeddings API.
- Embeddings and metadata are stored in a **persistent ChromaDB collection** (~62 MB), committed to the repository and rebuilt with a standalone builder script.
- A **lazy-loaded BM25 sparse index** is built at first query from the same ChromaDB documents for keyword-based retrieval alongside dense search.
- All retrieval artifacts are **immutable at query time**, enabling reproducible behavior across runs.
- No local embedding model is required at runtime — query embedding uses the same OpenAI API.

### 3. Query-time reasoning (agentic pipeline)

For each user query:

1. **Vagueness check** — Short queries without anatomical or symptom specificity are caught early and prompted for clarification. This check is **bypassed for follow-up questions** (when conversation history exists).
2. **Agentic query classification** assigns the query to a biomechanical category (benign, MSKNeurology-style syndrome, rare/serious, or unclear). Classification is **history-aware**: follow-up questions like "does it need surgery?" are classified based on the ongoing topic, not the query in isolation.
3. The query is **rewritten into biomechanics-aligned language** using conversation history to resolve pronouns and short follow-ups (e.g., "what about for TOS?" → full biomechanical query).
4. **Multi-query retrieval**: 2 alternative query reformulations are generated (via `gpt-4.1-nano`) and merged with the original to broaden retrieval coverage.
5. **Hybrid search** combines dense retrieval (ChromaDB) with sparse retrieval (BM25) using **Reciprocal Rank Fusion (RRF)** for robust ranking.
6. **Domain-specific heuristic biasing** adjusts distances to promote mechanism-dense sections (e.g., anatomy, biomechanics, assessment) and penalize narrative or low-yield content (e.g., case reports).  
   Long-form clinical text has structure that dense embeddings alone do not respect. Section headers, narrative vs. mechanism content, and article context matter. Heuristics encode these domain priors explicitly so failure modes remain inspectable.
7. An **optimized LLM-based reranker** (`gpt-4.1-nano`) reorders chunks *within each source article* rather than globally. Excerpts are truncated to 120 tokens and limited to 15 candidates for efficiency.
8. **Context compression** extracts only the most relevant sentences from each retrieved chunk using keyword-overlap scoring, reducing prompt size and focusing the LLM on pertinent information.
9. Retrieved chunks are **grouped by source**, prioritized by section, and **deterministically packed under a fixed token budget** (10,000 tokens), including controlled neighbor headroom.
10. A grounded answer is generated **strictly from the assembled context** using `gpt-4.1-mini`, with no external knowledge injection.

**Conversational follow-ups**: The system adapts its answer format based on conversation history. Initial clinical questions receive the **7-section biomechanical structure** (primary driver → neural consequences → muscular pattern → secondary effects → correction order → corrective emphasis → practical steps). Follow-up questions receive shorter, **direct conversational answers** without repeating the full structure.

<img alt="MSK Triage Chatbot — Welcome screen with region chips and clinical topic cards" src="docs/screenshots/welcome_screen.png" />

<img alt="MSK Triage Chatbot — Clinical topic categories (Pain Patterns, Functional & Activity)" src="docs/screenshots/clinical_topics.png" />

---

## Key design choices

- **Retrieval first, generation last:** The language model explains retrieved mechanisms; it does not invent them.
- **Hybrid retrieval:** Dense embeddings (semantic) + BM25 (keyword) fused via RRF for robust ranking across query types.
- **Multi-query expansion:** Alternative query reformulations broaden retrieval coverage for complex or nuanced questions.
- **Agentic retrieval alignment:** Queries are classified and rewritten to match the biomechanical language used in the corpus.
- **Heuristic biasing over opaque ranking:** Section priority, narrative penalties, and topic bonuses encode domain knowledge explicitly.
- **Deterministic context assembly:** Token budgets, per-source limits, and selection rules are fixed and inspectable.
- **Per-source reranking:** LLM reranking operates within articles to preserve topical coherence.
- **Conversational continuity:** Follow-up questions are interpreted in context, never triggering re-clarification for details already discussed.
- **Telemetry by default:** Retrieval confidence, timing, token usage, and selected sources are exposed in the UI.
- **No local model required:** All embedding and generation is handled by OpenAI APIs. No PyTorch, no GPU, under 200 MB runtime memory.
- **Reproducible by construction:** Immutable vector stores, fixed retrieval rules, and deterministic context packing yield identical behavior for identical inputs.


<img alt="MSK Triage Chatbot — 7-section biomechanical response with citations" src="docs/screenshots/conversation_response.png" />

<img alt="MSK Triage Chatbot — Telemetry panel showing retrieval stats and token usage" src="docs/screenshots/telemetry_panel.png" />


---

## Live deployment

The system is deployed as a split architecture for free-tier hosting:

```
Browser → Vercel (static frontend) → Render (FastAPI backend) → ChromaDB + OpenAI API
```

- **Frontend** (Vercel Free): Static HTML/CSS/JS chat interface at `frontend/`
  - 12 quick-start region chips + 16 categorized clinical topic cards with SVG icons
  - Real-time token streaming via SSE
  - Telemetry display (retrieval confidence, source count, latency, token usage)
- **Backend** (Render Free): FastAPI app at `backend/main.py` — runs `qaEngine.agentic_run()` against the committed `chroma_store/`
  - Streaming endpoint (`/ask/stream`) for real-time token delivery via Server-Sent Events
  - Optional Supabase integration for conversation history persistence and user authentication (JWT)
- **Vector store**: ~62 MB persistent ChromaDB committed to the repository (3072-dim embeddings, no rebuild on deploy)
- **Embedding + generation**: OpenAI API only — no local models, no PyTorch, under 200 MB runtime memory

Safety controls (public-facing):
- Max 1000 characters per question
- Max 5 conversation history turns
- Max 1000 output tokens
- In-memory rate limit: 5 requests/min per IP

### Models used

| Purpose | Model | Why |
|---|---|---|
| Embedding | `text-embedding-3-large` (3072-dim) | Higher retrieval accuracy than 1536-dim small |
| Generation | `gpt-4.1-mini` | Strong reasoning, fast, cost-effective |
| Reranking | `gpt-4.1-nano` | Fastest/cheapest for batch scoring |
| Query rewriting | `gpt-4.1-nano` | Fast reformulation and classification |

### Deployment steps

1. **Render**: Connect GitHub repo → auto-detects `render.yaml` → set `OPENAI_API_KEY` env var → deploy  
2. **Vercel**: Connect GitHub repo → root directory: `frontend/` → framework: None → deploy  
3. Update `frontend/app.js` with the Render backend URL  
4. Both auto-deploy on `git push`

---

## Local setup 

This project is intended primarily as a **retrieval systems engineering case study**.  
Running it locally is optional and aimed at inspection rather than end-user deployment.

### Requirements
- Python 3.10+
- An OpenAI API key (`OPENAI_API_KEY`) in `.env`
- The committed `chroma_store/` (included in the repository)

### Run the FastAPI backend locally

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Run the Streamlit UI locally (development)

```bash
pip install -r requirements.txt
streamlit run chatbot/mskbot.py
```

### (Optional) Rebuild corpus artifacts from scratch

```bash
# 1. Create chunks.parquet from HTML articles
python Text_Extraction/textExtract.py

# 2. Rebuild chroma_store with OpenAI embeddings (text-embedding-3-large)
python scripts/rebuild_chroma_openai.py
```

### (Optional) Run retrieval evaluation

```bash
python scripts/run_eval.py
```

Evaluates the production pipeline against the gold set (`Eval/gold_set_merged_for_eval.jsonl`), measuring retrieval accuracy (NDCG, MRR, Hit@k).

---

## Repository structure

```text
msk_chat/
├── backend/
│   ├── main.py                   # FastAPI app (Render deployment) — streaming, auth, rate limiting
│   └── requirements.txt          # Production-only dependencies
│
├── frontend/
│   ├── index.html                # Chat UI — welcome screen with region chips and clinical topic cards
│   ├── app.js                    # Frontend logic — streaming, auth, telemetry display
│   ├── styles.css                # Dark theme styles
│   └── vercel.json               # Vercel config
│
├── chatbot/
│   └── mskbot.py                 # Streamlit UI (local development)
│
├── VectorDB/
│   ├── qaEngine.py               # Core RAG engine — retrieval, hybrid search, reranking, context compression, generation
│   ├── ChromaDB.py               # Vector store builder
│   └── retrieval.py              # Retrieval utilities
│
├── Text_Extraction/
│   └── textExtract.py            # HTML cleaning and token-aware chunking
│
├── Embedding/
│   └── embedding.py              # Embedding generation (offline, legacy)
│
├── Eval/
│   ├── gold_set_merged_for_eval.jsonl  # Gold set for retrieval evaluation
│   ├── build_goldset.py          # Gold set construction
│   ├── eval_gold.py              # Evaluation with topic-aware scoring
│   └── model_comparison.py       # Cross-model comparison
│
├── scripts/
│   ├── rebuild_chroma_openai.py  # Rebuild chroma_store with OpenAI text-embedding-3-large
│   └── run_eval.py               # Production pipeline evaluation script
│
├── MSKArticlesINDEX/
│   ├── chunks.parquet            # Chunk table (1,301 chunks with text and metadata)
│   └── mskneurology.com/         # Offline HTML mirror (20 articles)
│
├── chroma_store/                 # Persistent ChromaDB store (committed, ~62 MB, 3072-dim)
├── embeddings/                   # Embedding artifacts and model metadata
│
├── render.yaml                   # Render deployment blueprint
├── requirements.txt              # Full local dependencies (includes Streamlit)
└── README.md
```
