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
