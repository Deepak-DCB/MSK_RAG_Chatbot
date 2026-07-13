# Runtime Contracts

## Backend endpoints
- `GET /health`
- `POST /ask`
- `POST /ask/stream`
- `GET /history`

`GET /history` is authenticated and paginated with bounded `limit`/`offset` inputs.

## Streaming contract
`/ask/stream` emits:
- token events (`data: {"token": ...}`)
- final done event with metadata, citations, and `complete: true|false`

On failed or incomplete streaming outcomes, the done payload includes `error`, `request_id`, and `complete: false` for log correlation.

## Provider timeout and cancellation
- OpenAI and OpenAI-compatible provider clients use `OPENAI_TIMEOUT_SECONDS`, defaulting to 90 seconds per HTTP request. The setting covers embeddings, generation, and configured fallback providers.
- The stream’s existing 60-second idle and 120-second total deadlines remain the user-facing limits. A disconnected or expired stream cancels cooperatively at the next token and reaps its worker; a provider blocked before its first token can remain active until the 90-second client timeout because Python cannot safely kill that provider thread.
- The tradeoff is bounded provider/resource time versus long responses: a request that needs more than 90 seconds for one provider may degrade to the normal fallback ladder or deterministic evidence-only answer. Once a stream has emitted partial answer text, it does not switch providers; it ends with `complete: false` so the UI cannot present a mixed answer as complete.

## Required metadata fields
- retrieval and generation timing
- token usage counts
- retrieval confidence
- category and refined query
- reranker mode and config source

## Safety boundaries
- Question length cap
- History turn cap
- Output token cap
- Per-IP rate limiting
- Request config allowlist for public overrides
