# MSK RAG Chatbot  
### Mechanism-Level Clinical Question Answering over MSK Neurology (Retrieval-Augmented Generation)

## TL;DR

- Domain-constrained RAG system for musculoskeletal neurology and biomechanics  
- Retrieval-first design with deterministic context assembly and explicit heuristics  
- Agentic query classification and rewrite to align user language with biomechanical mechanisms  
- No fine-tuning, no end-to-end black box; emphasis on inspectability and failure analysis  

This repository implements a **retrieval-augmented question answering (RAG) system** for answering **mechanism-level clinical questions** grounded in a corpus derived from **MSKNeurology.com** (Kjetil Larsen).

Rather than treating the language model as an end-to-end reasoning engine, the system treats **answer quality as a downstream effect of retrieval quality**, and is designed to expose, constrain, and debug each step of the retrieval and context-selection process.

The system surfaces retrieved chunks, distances, heuristic adjustments, reranking behavior, token budgets, latency, and confidence signals so that outputs can be **inspected, audited, and failure-mode analyzed**, not merely consumed.

> **Note:** This system is not a medical device and does not provide diagnoses or treatment recommendations. It is an educational and research-oriented explainer grounded strictly in retrieved corpus content.

---

## Motivation

Musculoskeletal neurology is a narrow domain where valid explanations depend on **anatomy, biomechanics, and neurovascular space**, typically described in long-form clinical articles rather than structured knowledge bases.

General-purpose language models frequently hallucinate or over-generalize in this domain when used without strong retrieval constraints, particularly when asked “why” or “mechanism” questions.

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

- Dense embeddings are generated for all chunks using a **SentenceTransformers-compatible embedding model** with L2-normalized vectors.
- Embeddings and metadata are stored in a **persistent ChromaDB collection**, rebuilt explicitly via a standalone builder script.
- All retrieval artifacts are **immutable at query time**, enabling reproducible behavior across runs.

### 3. Query-time reasoning

For each user query:

1. **Agentic query classification** assigns the query to a biomechanical category (benign, MSKNeurology-style syndrome, rare/serious, or unclear).
2. The query is **rewritten into biomechanics-aligned language** to improve dense retrieval alignment with the corpus.
3. Dense retrieval is performed against the persistent vector store.
4. **Domain-specific heuristic biasing** adjusts distances to promote mechanism-dense sections (e.g., anatomy, biomechanics, assessment) and penalize narrative or low-yield content (e.g., case reports).
5. An **optional LLM-based reranker** reorders chunks *within each source article* rather than globally.
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
- **CPU-only execution:** The system is designed to run locally without GPUs or specialized hardware; LLM usage is limited to query-time reasoning.

<img width="2879" height="1799" alt="Streamlit UI with retrieval telemetry" src="https://github.com/user-attachments/assets/a5cf6d57-edfe-41cc-a4ae-5779213506d7" />

## Repository structure

```text
msk_chat/
├── chatbot/
│   └── mskbot.py                 # Streamlit UI and interaction layer
│
├── Text_Extraction/
│   └── textExtract.py            # HTML cleaning and token-aware chunking
│
├── Embedding/
│   └── embedding.py              # Embedding generation
│
├── VectorDB/
│   ├── ChromaDB.py               # Persistent vector store construction/loading
│   ├── qaEngine.py               # Core RAG logic (retrieval, biasing, reranking, packing)
│   └── retrieval.py              # Retrieval utilities
│
├── MSKArticlesINDEX/
│   ├── all_articles.jsonl        # Article-level metadata
│   ├── chunks.parquet            # Chunk table with text and metadata
│   └── mskneurology.com/         # Offline HTML mirror
│
├── embeddings/                   # Generated embedding artifacts
├── chroma_store/                 # Persistent ChromaDB store
│
├── Eval/                         # Evaluation scripts and plots
├── Evaluation/                   # Metric histories across runs
│
├── reviewGoldset.py              # Gold set inspection and annotation UI
└── chunk_editor.py               # Chunk repair 
