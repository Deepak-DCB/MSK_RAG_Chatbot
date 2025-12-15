# MSK RAG Chatbot  
### Clinical Question Answering over MSK Neurology (Retrieval-Augmented Generation)

This repository contains a retrieval-augmented question answering (RAG) system designed to answer **mechanism-level clinical questions** using a corpus derived from **MSKNeurology.com** (Kjetil Larsen). The system emphasizes **retrieval correctness, transparency, and interpretability** over end-to-end generation.

The system exposes the full retrieval process—selected chunks, similarity scores, heuristic biasing, reranking behavior, token budgeting, and latency—so that answers can be inspected, evaluated, and debugged.

> **Scope note:** This system is not a medical device and does not provide diagnoses or treatment recommendations. It is an educational and research-oriented "explainer" grounded strictly in retrieved corpus content.

---

## Motivation

Musculoskeletal neurology is a narrow domain where valid explanations depend on anatomy, biomechanics, and neurovascular relationships described in long-form clinical articles rather than structured knowledge bases. General-purpose language models frequently hallucinate or overgeneralize when answering such questions without explicit grounding.

This project explores how far **carefully designed retrieval pipelines**—rather than increasingly complex prompting—can improve answer quality, traceability, and failure analysis in a specialized clinical corpus.

---

## System overview

The system is structured as a **three-stage pipeline**: offline corpus processing, persistent retrieval infrastructure, and online query-time reasoning.

### 1. Offline corpus processing
- HTML pages from MSKNeurology.com are mirrored locally.
- Text is cleaned and segmented using sentence-first, token-aware chunking.
- Each chunk is annotated with article, section, and positional metadata.
- Outputs are persisted as a structured chunk table.

### 2. Persistent retrieval infrastructure
- Dense embeddings are generated for all chunks using a SentenceTransformers-compatible model.
- Embeddings and metadata are stored in a persistent ChromaDB collection.
- All retrieval artifacts are immutable at query time, enabling reproducible behavior.

### 3. Query-time reasoning
For each user query:
1. An LLM classifies the query and optionally rewrites it into biomechanics-aligned language.
2. Dense retrieval is performed against the vector store.
3. MSK-specific heuristic biasing promotes mechanism-dense sections and demotes low-yield content.
4. An optional LLM reranker reorders chunks within each retrieved article.
5. Retrieved context is deterministically packed under a fixed token budget.
6. A grounded answer is generated strictly from the assembled context.

<img width="6044" height="3124" alt="MSKRAGArch drawio" src="https://github.com/user-attachments/assets/2b376e20-653e-4885-b228-b4ec330d98f0" />

---

## Key design choices

- **Retrieval first, generation last:** Answer quality is a downstream effect of retrieval quality.
- **Deterministic context assembly:** Token budgets and selection rules are fixed and inspectable.
- **Heuristic biasing over black-box ranking:** Domain knowledge is encoded explicitly rather than deferred to a model.
- **Telemetry by default:** Retrieval behavior is surfaced in the UI to enable qualitative analysis.
- **CPU-only operation:** The system is designed to run on a single machine without specialized hardware.

<img width="2879" height="1799" alt="image" src="https://github.com/user-attachments/assets/a5cf6d57-edfe-41cc-a4ae-5779213506d7" />

---

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
