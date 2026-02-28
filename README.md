# MSK RAG Chatbot  
### Biomechanics Clinical Question Answering using MSK Neurology (Retrieval-Augmented Generation)

## TL;DR

**This is:** A retrieval systems engineering case study in a constrained clinical domain.  
**This is not:** A diagnostic tool, autonomous clinician, or end-to-end learned medical model.

- Domain-constrained RAG system for musculoskeletal neurology and biomechanics  
- Retrieval-first design with deterministic context assembly and explicit heuristics  
- Agentic query classification and rewrite to align user language with biomechanical mechanisms  
- No fine-tuning, no end-to-end black box; emphasis on inspectability and failure analysis



This repository implements a **retrieval-augmented question answering (RAG) system** for answering **mechanism-level clinical questions** grounded in a corpus derived from **MSKNeurology.com** (Kjetil Larsen).

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

- Dense embeddings are generated for all chunks using **OpenAI `text-embedding-3-small`** (1536-dim), called via the embeddings API.
- Embeddings and metadata are stored in a **persistent ChromaDB collection** (~29 MB), committed to the repository and rebuilt with a standalone builder script.
- All retrieval artifacts are **immutable at query time**, enabling reproducible behavior across runs.
- No local embedding model is required at runtime — query embedding uses the same OpenAI API.

### 3. Query-time reasoning

For each user query:

1. **Agentic query classification** assigns the query to a biomechanical category (benign, MSKNeurology-style syndrome, rare/serious, or unclear).
2. The query is **rewritten into biomechanics-aligned language** to improve dense retrieval alignment with the corpus.
3. Dense retrieval is performed against the persistent vector store.
4. **Domain-specific heuristic biasing** adjusts distances to promote mechanism-dense sections (e.g., anatomy, biomechanics, assessment) and penalize narrative or low-yield content (e.g., case reports).  
   Long-form clinical text has structure that dense embeddings alone do not respect. Section headers, narrative vs. mechanism content, and article context matter. Heuristics encode these domain priors explicitly so failure modes remain inspectable.

  
5. An **optional LLM-based reranker** reorders chunks *within each source article* rather than globally. Reranking is deliberately constrained to preserve article coherence.
6. Retrieved chunks are **grouped by source**, prioritized by section, and **deterministically packed under a fixed token budget**, including controlled neighbor headroom.
7. A grounded answer is generated **strictly from the assembled context**, with no external knowledge injection.

<img width="6044" height="3124" alt="MSK RAG architecture diagram" src="https://github.com/user-attachments/assets/2b376e20-653e-4885-b228-b4ec330d98f0" />

---

## Key design choices

- **Retrieval first, generation last:** The language model explains retrieved mechanisms; it does not invent them.
- **Agentic retrieval alignment:** Queries are classified and rewritten to match the biomechanical language used in the corpus.
- **Heuristic biasing over opaque ranking:** Section priority, narrative penalties, and topic bonuses encode domain knowledge explicitly.
- **Deterministic context assembly:** Token budgets, per-source limits, and selection rules are fixed and inspectable.
- **Per-source reranking:** Optional LLM reranking operates within articles to preserve topical coherence.
- **Telemetry by default:** Retrieval confidence, timing, token usage, and selected sources are exposed in the UI.
- **No local model required:** All embedding and generation is handled by OpenAI APIs. No PyTorch, no GPU, under 200 MB runtime memory.
- **Reproducible by construction:** Immutable vector stores, fixed retrieval rules, and deterministic context packing yield identical behavior for identical inputs.


<img width="2879" height="1799" alt="Streamlit UI with retrieval telemetry" src="https://github.com/user-attachments/assets/a5cf6d57-edfe-41cc-a4ae-5779213506d7" />


---

## Live deployment

The system is deployed as a split architecture for free-tier hosting:

```
Browser → Vercel (static frontend) → Render (FastAPI backend) → ChromaDB + OpenAI API
```

- **Frontend** (Vercel Free): Static HTML/CSS/JS chat interface at `frontend/`
- **Backend** (Render Free): FastAPI app at `backend/main.py` — runs `qaEngine.agentic_run()` against the committed `chroma_store/`
- **Vector store**: ~29 MB persistent ChromaDB committed to the repository (no rebuild on deploy)
- **Embedding + generation**: OpenAI API only — no local models, no PyTorch, under 200 MB runtime memory

Safety controls (public-facing):
- Max 1000 characters per question
- Max 5 conversation history turns
- Max 1000 output tokens
- In-memory rate limit: 5 requests/min per IP

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

# 2. Rebuild chroma_store with OpenAI embeddings
python scripts/rebuild_chroma_openai.py
```

---

## Repository structure

```text
msk_chat/
├── backend/
│   ├── main.py                   # FastAPI app (Render deployment)
│   └── requirements.txt          # Production-only dependencies
│
├── frontend/
│   ├── index.html                # Chat UI
│   ├── app.js                    # Frontend logic
│   ├── styles.css                # Dark theme styles
│   └── vercel.json               # Vercel config
│
├── chatbot/
│   └── mskbot.py                 # Streamlit UI (local development)
│
├── VectorDB/
│   ├── qaEngine.py               # Core RAG logic (retrieval, biasing, reranking, packing)
│   ├── ChromaDB.py               # Vector store builder
│   └── retrieval.py              # Retrieval utilities
│
├── Text_Extraction/
│   └── textExtract.py            # HTML cleaning and token-aware chunking
│
├── Embedding/
│   └── embedding.py              # Embedding generation (offline)
│
├── scripts/
│   └── rebuild_chroma_openai.py  # One-time script: rebuild chroma with OpenAI embeddings
│
├── MSKArticlesINDEX/
│   ├── chunks.parquet            # Chunk table with text and metadata
│   └── mskneurology.com/         # Offline HTML mirror
│
├── chroma_store/                 # Persistent ChromaDB store (committed, ~29 MB)
├── embeddings/                   # Embedding artifacts and model metadata
│
├── render.yaml                   # Render deployment blueprint
├── requirements.txt              # Full local dependencies (includes Streamlit)
└── README.md
