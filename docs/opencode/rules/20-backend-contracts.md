# Backend Contracts

## Endpoint scope
- `/health`: service and collection readiness
- `/ask`: non-streaming answer + telemetry
- `/ask/stream`: SSE token stream + final metadata payload
- `/history`: authenticated conversation history access

## Safety caps
- Question length cap
- History turn cap
- Output token cap
- IP rate limiting

These caps must remain enforced server-side.

## Telemetry contract
Preserve response fields used by frontend and eval flows, including:
- retrieval and generation timing
- token counts
- retrieval confidence
- category and rewritten query fields
- reranker mode and applied config metadata

Any telemetry shape change must be coordinated with frontend updates.
