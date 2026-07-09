# Architecture Context

## System shape
- Frontend: static Vercel app (`frontend/`)
- Backend: FastAPI service on Render (`backend/main.py`)
- Retrieval core: `VectorDB/qaEngine.py`
- Data store: committed Chroma collection in `chroma_store/`
- Model APIs: OpenAI embeddings + generation/reranking models

## Query flow summary
1. Frontend sends question/history/config to `/ask/stream`.
2. Backend runs `agentic_run`.
3. Retrieval and ranking build context.
4. Answer streams token-by-token via SSE.
5. Final telemetry and citations are returned in done event.

## Design priorities
- Retrieval-first, evidence-grounded reasoning.
- Deterministic and inspectable ranking/context packing.
- Conservative triage behavior over speculative output.
