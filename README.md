# MSK RAG Chatbot — Clinical QA over MSK Neurology (RAG + Telemetry)

A local, end-to-end **retrieval-augmented generation (RAG)** system for asking clinical “mechanism” questions against an offline mirror of **MSKNeurology.com** (Kjetil Larsen). The system is built for **transparent retrieval**: it shows the chunks it used, similarity scores, biasing, reranker outputs, token budgeting, and latency telemetry, and then generates a grounded answer using the retrieved context.

> **Important scope note:** This project is **not** a medical device and does **not** provide diagnoses or prescriptions. It is a mechanism-focused, evidence-grounded explainer that depends on the supplied corpus context.

---

## What this project does

From the report, the pipeline is:

1. **Offline corpus**: HTTrack mirror of MSKNeurology.com (local HTML files).
2. **Chunking**: sentence-first, token-aware chunking into traceable units with article/section metadata.
3. **Embeddings**: dense embeddings (SentenceTransformers-compatible model).
4. **Vector store**: persistent **ChromaDB** collection with metadata.
5. **Agentic retrieval control**: LLM-based **query classification + biomechanics-focused query rewriting** before dense search.
6. **Retrieval**: dense search + MSK-specific heuristic biasing + optional LLM reranking (intra-article).
7. **Context packing**: deterministic packing under a token budget, with optional neighbor chunks and optional gated conversation memory.
8. **Answering**: OpenAI chat completions over the assembled context.
9. **UI**: Streamlit chat UI with telemetry and controls.

(These components are described in the submitted report.)

---

## Live demo

Because this app requires a server (Python + retrieval + OpenAI calls), it **cannot run on GitHub Pages** (Pages is static hosting only). A typical “portfolio” setup is:

- **Streamlit Community Cloud** for the interactive app
- **GitHub Pages** for a static landing page (screenshots, architecture, and links)

- **Streamlit app:** `https://<your-app>.streamlit.app`
- **Demo video:** `https://<youtube-or-drive-link>`
- **Project report PDF:** `./MSK_RAG_Report.pdf` 

---
## Repository layout

```text
msk_chat/
├── .env
│   └── Local environment variables (API keys; never committed)
├── .gitignore
│   └── Excludes secrets, embeddings, vector DB, and large artifacts
├── app.py
│   └── Optional top-level launcher / experiments
│
├── chatbot/
│   └── mskbot.py
│       └── Streamlit UI (input, answers, retrieval telemetry)
│
├── Text_Extraction/
│   └── textExtract.py
│       └── HTML cleaning and sentence/token-aware chunking
│
├── Embedding/
│   └── embedding.py
│       └── Chunk embedding generation (SentenceTransformers)
│
├── VectorDB/
│   ├── ChromaDB.py
│   │   └── Build/load persistent ChromaDB store
│   ├── qaEngine.py
│   │   └── Core RAG logic (retrieval, biasing, rerank, context pack)
│   └── retrieval.py
│       └── Retrieval utilities and filters
│
├── MSKArticlesINDEX/
│   ├── all_articles.jsonl
│   │   └── Normalized article metadata
│   ├── chunks.parquet
│   │   └── Chunk table (text + metadata)
│   └── mskneurology.com/
│       └── Mirrored source HTML (HTTrack)
│
├── embeddings/
│   ├── embeddings.npy
│   │   └── Text chunk embeddings (generated)
│   ├── embeddingsImages.npy
│   │   └── Optional image embeddings (generated)
│   └── embedding_model.txt
│       └── Embedding model identifier
│
├── chroma_store/
│   ├── chroma.sqlite3
│   │   └── ChromaDB metadata
│   └── <collection-id>/
│       └── Vector index binaries (generated)
│
├── Eval/
│   ├── build_goldset.py
│   │   └── Gold set construction
│   ├── eval_gold.py
│   │   └── Baseline retrieval evaluation
│   ├── eval_gold_reranked.py
│   │   └── Evaluation with LLM reranking
│   └── plots / csv / json
│       └── Evaluation artifacts
│
├── Evaluation/
│   └── *.csv
│       └── Metric histories across runs
│
├── goldset_ui/
│   ├── app.py
│   │   └── Gold set inspection UI
│   ├── templates/
│   ├── static/
│   └── gold_utils/
│
├── runAll.py
│   └── End-to-end pipeline runner
├── chunk_editor.py
│   └── Chunk inspection and repair tools
│
└── paper.tex
    └── LaTeX source of accompanying report
```



---

## Quickstart (local)

### 0) Prerequisites
- **Python 3.12** recommended (matches the report’s environment).
- Enough disk space for the corpus mirror + embeddings + Chroma store.

### 1) Clone + create a virtual environment
```bash
git clone https://github.com/Deepak-DCB/MSK_RAG_Chatbot.git
cd MSK_RAG_Chatbot

# If your code is inside /msk_chat, do:
# cd msk_chat

python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies
If you already have a `requirements.txt`, use it:
```bash
pip install -r requirements.txt
```

If you *don’t* yet have one, generate it from your working environment:
```bash
pip freeze > requirements.txt
```

Typical packages used in this project include:
- `streamlit`, `openai`, `python-dotenv`
- `chromadb`
- `sentence-transformers`, `torch` (CPU is fine)
- `pandas`, `numpy`
- `beautifulsoup4`, `lxml`
- `nltk`
- `tiktoken` (optional but useful)

### 3) Configure your OpenAI API key (never commit this)
Create a `.env` file in the **project root**:

```bash
OPENAI_API_KEY=your_key_here
```

Do **not** hardcode your key in code or commit `.env`.

### 4) Run the Streamlit app
From the project root:

```bash
streamlit run chatbot/mskbot.py
```

If your entrypoint differs (some repos use `app.py`), run that instead:
```bash
streamlit run app.py
```

---

## Rebuilding the data pipeline (from an HTTrack mirror)

You have two “portfolio modes”:

### Mode A — Ship only code (recommended for public GitHub)
- Keep the HTTrack mirror **out of the repo** (copyright + size).
- Keep `embeddings/` and `chroma_store/` **out of the repo** (size).
- Provide scripts + instructions so others can rebuild locally.

### Mode B — Ship a prebuilt index (convenient but heavy)
- Include `chroma_store/` + `embeddings/` so the app runs immediately.
- This can blow up repo size quickly. Consider **Git LFS** (Large File Storage) or release assets.

---

### Step 1) Chunk the corpus (HTML → chunks.parquet)

Input: local mirror folder (HTTrack output), e.g. `MSKArticlesINDEX/mskneurology.com/`  
Output: `MSKArticlesINDEX/all_articles.jsonl` and `MSKArticlesINDEX/chunks.parquet`

Run (example):
```bash
python textExtract.py \
  --root MSKArticlesINDEX/mskneurology.com \
  --out  MSKArticlesINDEX
```

If your `textExtract.py` lives somewhere else, adjust the path.

### Step 2) Embed chunks (chunks.parquet → embeddings.npy)
This produces:
- `embeddings/embeddings.npy`
- `embeddings/embedding_model.txt` (used later for reproducibility)

Run (example):
```bash
python Embedding/embedding.py \
  --input MSKArticlesINDEX/chunks.parquet \
  --outdir embeddings \
  --model mixedbread-ai/mxbai-embed-large-v1
```

### Step 3) Build the ChromaDB store (embeddings → chroma_store)
Run (example):
```bash
python ChromaDB.py \
  --chunks MSKArticlesINDEX/chunks.parquet \
  --embeds embeddings/embeddings.npy \
  --store chroma_store
```

After this, the Streamlit app and QA engine should load the persistent store from `chroma_store/`.

---

## Deployment (Streamlit Community Cloud)

This is the easiest “resume-friendly” deployment: a public repo + a hosted Streamlit app.

### 1) Prep your repo for Streamlit
You need at minimum:
- `requirements.txt`
- A clear entrypoint: `chatbot/mskbot.py` or `app.py`

If your repo root is not the runtime root (for example, code is inside `msk_chat/`), you have two options:
- Move project files to repo root, **or**
- Set Streamlit’s “Main file path” to `msk_chat/chatbot/mskbot.py`

### 2) Add your API key as a Streamlit secret
In Streamlit Cloud:
- App → Settings → Secrets
- Add:

```toml
OPENAI_API_KEY="your_key_here"
```

Then, in code, read it from environment (`os.environ["OPENAI_API_KEY"]`) or `python-dotenv` locally.

### 3) Large files: decide what you’re hosting
Streamlit Cloud has limits; shipping a big `chroma_store/` may fail.

Recommended approach:
- Keep `chroma_store/` out of the repo.
- Provide a small “sample” index for demo, or rebuild on first run (slow).
- Alternatively: host the index as a downloadable release asset and fetch it at startup (more engineering).

---

## GitHub Pages (optional static site)

GitHub Pages cannot run the chatbot, but it *can* host a clean landing page:

- `docs/index.md` with:
  - screenshots
  - architecture diagram
  - evaluation plot
  - links to Streamlit app + report PDF + demo video

To enable:
1. Create a `docs/` folder
2. Add a minimal `docs/index.md`
3. Repo Settings → Pages → deploy from `main` branch, `/docs`


## Making the repo public-friendly (recommended)

If you want hiring managers to be able to clone quickly and run safely, keep large/generated and rights-sensitive assets out of git history.

### Suggested `.gitignore`
Add (or confirm) entries like:

```gitignore
# Secrets
.env
.streamlit/secrets.toml

# Python
.venv/
__pycache__/
*.pyc

# Large generated artifacts
chroma_store/
embeddings/
MSKArticlesINDEX/mskneurology.com/

# OS/editor noise
.DS_Store
.vscode/
.idea/
```

### “Two-repo” pattern (cleanest)
- **Public repo**: code + docs + small screenshots + evaluation plots
- **Private repo / local**: full HTTrack mirror + embeddings + chroma_store

You can also publish a **small sample index** for demo purposes (e.g., 1–2 articles) so the app is runnable without distributing the full corpus.


---

## Evaluation

The report evaluates retrieval on a 50-question gold set using:
- **Hit@K**
- **MAP@5**
- **NDCG@5**
- **Mean Reciprocal Rank (MRR)**

It compares:
- **Baseline**: dense retrieval + agentic rewrite + heuristic biasing (no LLM rerank)
- **Full**: same + LLM reranker

Result summary (as stated in the report): the reranked pipeline underperformed the baseline across the reported retrieval metrics, likely because reranking occurs late and only reorders chunks within already-selected articles.

See:
- `Eval/` scripts and `eval_plot.png`
- `eval_results.json` and `eval_results_topicaware.json`

---

## Security + secrets

- Never commit `.env`
- Never print API keys into logs
- For Streamlit Cloud, use **Secrets**
- For GitHub Actions, use **Repository Secrets** (if you add CI that needs keys — ideally avoid)

---

## Common troubleshooting

### “Chroma store not found”
- Ensure `chroma_store/` exists at the expected path.
- If your code assumes a specific project root, run from that directory.

### “Embedding model downloads slowly / fails”
- First-time SentenceTransformers model download can be slow.
- On restricted environments, pre-download and cache the model.

### “Token counting errors”
- `tiktoken` is optional; if missing, some scripts fall back to word-to-token approximations.

### “Streamlit Cloud is too slow”
- CPU-only inference is fine, but indexing and large stores can be heavy.
- Reduce `top_k`, reduce reranker usage, keep token budgets reasonable.

---

## Attribution and rights

- The corpus content originates from **MSKNeurology.com** (Kjetil Larsen).  
  If you publish mirrors or large excerpts, confirm you have rights to do so.
- This repository is intended as an educational/engineering portfolio piece.

---

## Citation

If you reference this system in writing:

> Binkam, D. (2025). *A Retrieval-Augmented Clinical Question Answering System for Musculoskeletal Neurology.* Towson University, COSC 880 Graduate Project.

(You can add a BibTeX block here if you want.)

---

## License

Choose a license that matches your intent (MIT is common for portfolio code).  
If you include any mirrored site content, ensure the license/rights situation is compatible.

