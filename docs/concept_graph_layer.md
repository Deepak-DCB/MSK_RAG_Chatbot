# Concept Graph Layer

The concept graph layer is an optional, evidence-grounded mechanism map built on top of the existing hierarchical RAG artifacts. It does not replace Chroma retrieval, hybrid BM25/dense retrieval, safety gates, query rewriting, or the hierarchical article/section/evidence-span layer.

## Purpose

The hierarchy answers where text came from: article, section, paragraph, and evidence span.

The graph answers how detected concepts connect: structures, spaces, symptoms, posture patterns, movement patterns, mechanisms, tests, and safety concepts.

The graph is designed to focus context, not enlarge prompts. When useful mechanism paths are found, the runtime can prefer compact graph paths and their strongest supporting evidence spans instead of dumping broad sections or full articles.

## Artifacts

Graph artifacts live under `MSKArticlesINDEX/graph/`:

- `nodes.jsonl`
- `edges.jsonl`
- `paths.jsonl`
- `claims.jsonl`
- `graph_manifest.json`

Build them with:

```bash
python scripts/build_concept_graph.py
```

The first builder is deterministic and requires no paid LLM calls. It reads:

- `MSKArticlesINDEX/hierarchical/evidence_spans.jsonl`
- `MSKArticlesINDEX/hierarchical/sections.jsonl`
- `MSKArticlesINDEX/hierarchical/articles.jsonl`

## Extraction

`VectorDB/graph_vocab.py` defines canonical entities and aliases for the current corpus. The builder scans evidence spans, detects aliases, creates nodes, creates weak co-mention edges, and promotes relation edges only when conservative text patterns support them.

Uncertain relation language is intentionally conservative. For example, `may cause`, `can cause`, and `leads to` are usually represented as `may_produce` or `may_contribute_to`, not direct causation.

## Support Labels

Edges carry:

- `support_level`: `direct`, `indirect`, `inferred_from_same_section`, `inferred_from_path`, `weak`, or `unsupported`
- `claim_strength`: `strong`, `moderate`, `weak`, or `speculative`
- `clinical_risk`: `low`, `medium`, or `high`
- source span, section, and article IDs where available

Direct support means the edge is supported by a detected evidence span. Same-section bridges are weaker and are used only to preserve reviewable mechanism continuity without inventing unsupported direct claims.

## Paths

`VectorDB/graph_paths.py` builds initial mechanism path families when intermediate edges exist. It does not force complete paths when evidence is missing.

Long chains remain multi-step. For example, a path may preserve:

```text
scapular depression -> clavicle -> costoclavicular space -> brachial plexus -> neuralgia
```

The assistant should not collapse that into `scapular depression causes numbness`. If a path contains weak or inferred edges, it should be explained as an indirect possible mechanism and the weakest step should remain visible.

## Runtime Use

`VectorDB/graph_retrieval.py` provides deterministic helpers:

- `load_graph()`
- `find_nodes(query)`
- `find_edges_for_node(node_id)`
- `find_paths_for_nodes(node_ids)`
- `get_supporting_spans_for_path(path_id)`
- `build_graph_context(...)`
- `format_graph_context(...)`

`VectorDB/qaEngine.py` adds optional config fields:

- `use_graph_context=True`
- `graph_context_strategy="mechanism_paths"`
- `graph_focus_context=True`
- graph path, edge, span, and token caps

If graph artifacts are missing, the engine falls back safely and records `graph_available=false` plus `graph_fallback_reason`.

## Token Budget Behavior

Graph context has its own compact budget, defaulting to `graph_max_tokens=1800`. With `graph_focus_context=true` and useful paths present, `hybrid_long_context` avoids broad section/article expansion and keeps hierarchy anchors, evidence spans, selected chunks, and compact graph context.

This is intended to reduce or preserve prompt size relative to broad `hybrid_long_context`, not add a second large context layer.

## Safety Boundaries

The graph is not diagnostic. It can represent possible mechanism chains, but it cannot make unsupported clinical conclusions.

The prompt instructs generation to:

- use graph context only when supported by evidence spans
- avoid turning indirect paths into direct causal claims
- preserve safety escalation for numbness, weakness, vascular symptoms, or progressive neurologic symptoms
- avoid diagnosis and treatment prescriptions beyond existing system boundaries

## Limitations

- The first graph is rule-based and alias-dependent.
- Co-mention edges are weak by design.
- Same-section bridge edges help path continuity but should be treated as indirect.
- There is no graph vector search yet.
- The graph uses only the current corpus and does not add external medical sources.
- The graph is not corpus-complete. The coverage audit in `docs/opencode/context/concept-graph-coverage-audit.md` found uneven article coverage and major gaps outside TOS/scapular/neck mechanisms.
- Newly added non-TOS path families are review aids, not validated clinical pathways; paths with weak edges must stay qualified as possible indirect mechanisms.

## Future Work

- Add graph-specific retrieval evaluation with expected path/edge labels.
- Add reviewer tooling for correcting edge labels and support strength.
- Add optional graph vector search after deterministic behavior is measured.
- Tune path selection against prompt-token and answer-grounding metrics.
- Add clinician review of high-risk graph paths before exposing stronger path language.
