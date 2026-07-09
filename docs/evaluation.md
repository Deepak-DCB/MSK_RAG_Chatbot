# Evaluation and Evidence

This repository separates evaluation into three layers:

1. `retrieval relevance` - whether the right source and chunk show up.
2. `answer grounding` - whether the answer cites the required source and matches lightweight claim checks.
3. `safety / triage behavior` - whether urgent cases are escalated and false reassurance is avoided.

The canonical runner is `scripts/run_eval_production.py`. It writes reproducible artifacts to `Evaluation/runs/<run_id>/`:

- `cases.jsonl`
- `run_report.json`
- `run_notes.md`

## Current checked-in retrieval evidence

The repository already includes two legacy ablation outputs:

- `eval_results_topicaware.json`
- `eval_results_topicaware_reranked.json`

They cover a 50-question topic-aware gold set from the earlier retrieval harness. The summary below is computed directly from those checked-in files.

| Variant | Evidence file | Cases | Hit@1 article | Hit@5 chunk | MRR article | MRR chunk | NDCG@5 | Readout |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Topic-aware baseline | `eval_results_topicaware.json` | 50 | 98.0% | 94.0% | 0.990 | 0.762 | 0.787 | Strong retrieval baseline |
| Topic-aware + per-source reranker | `eval_results_topicaware_reranked.json` | 50 | 60.0% | 38.0% | 0.722 | 0.281 | 0.273 | Negative result; reranker degraded ranking |

This negative result is intentionally kept visible. The repo is stronger if it shows measured ablations honestly instead of implying every added component helped.

Regenerate that table with:

```bash
python scripts/summarize_eval_results.py --format markdown
```

## Canonical production-faithful runner

Dry-run the current production harness first:

```bash
python scripts/run_eval_production.py --dry-run --max-cases 5
```

Run the production-faithful retrieval evaluation on the gold set:

```bash
python scripts/run_eval_production.py --max-cases 10 --price-input-per-1k 0.0 --price-output-per-1k 0.0
```

Key properties:

- calls `agentic_run()` instead of a toy retrieval shortcut
- records commit hash, dataset hash, model config, and pipeline mode
- computes retrieval metrics when the dataset includes gold labels
- emits `not_evaluated` instead of fake zeros for layers not measured

## Answer-level and safety datasets

The same runner now detects additional dataset types and turns on the corresponding checks:

### Citation / grounding checks

Dataset: `datasets/citation-tests.jsonl`

```bash
python scripts/run_eval_production.py --dataset datasets/citation-tests.jsonl --max-cases 3
```

What it measures today:

- required source cited rate
- required source present in retrieved context
- rule-based claim-support label match rate

What it does not yet measure fully:

- semantic claim extraction across arbitrary answers
- contradiction detection beyond the current rule-based proxy

### Red-flag safety checks

Dataset: `datasets/red-flag-cases.jsonl`

```bash
python scripts/run_eval_production.py --dataset datasets/red-flag-cases.jsonl --dry-run --max-cases 50
```

What it measures today:

- red-flag escalation recall
- red-flag escalation precision
- false reassurance rate
- count of critical safety failures

The red-flag dataset now includes urgent cases and non-urgent controls so deterministic safety-gate behavior can be checked for both missed escalation and over-escalation.

### Vague-query clarification checks

Dataset: `datasets/vague-query-cases.jsonl`

```bash
python scripts/run_eval_production.py --dataset datasets/vague-query-cases.jsonl --dry-run --max-cases 25
```

What it checks locally:

- whether underspecified prompts request location, sensation, and trigger/timing
- whether adaptive clarification reminds users to mention urgent signs
- whether specific prompts continue into normal educational triage instead of forced intake

### Product-boundary checks

Datasets:

- `datasets/off-topic-cases.jsonl`
- `datasets/multi-turn-cases.jsonl`
- `datasets/unsupported-claim-cases.jsonl`

```bash
python scripts/run_eval_production.py --dataset datasets/off-topic-cases.jsonl --dry-run --max-cases 25
python scripts/run_eval_production.py --dataset datasets/multi-turn-cases.jsonl --dry-run --max-cases 10
python scripts/run_eval_production.py --dataset datasets/unsupported-claim-cases.jsonl --dry-run --max-cases 10
```

What the zero-cost `product_behavior` layer checks locally:

- deterministic safety-gate trigger expectations
- clarification versus continue behavior
- non-MSK scope boundaries
- medication and diagnosis boundary responses
- no diagnosis-certainty language in boundary responses
- no medication/dose/treatment prescription in boundary responses

This layer only evaluates deterministic local gates and static response boundaries. It does not measure semantic answer correctness for normal RAG answers. Dry-run case outputs include `response_source`; answer-text boundary checks stay `not_evaluated` when no local response was generated.

### Triage answer-quality checks

Dataset: `datasets/triage-cases.jsonl`

```bash
python scripts/run_eval_production.py --dataset datasets/triage-cases.jsonl --max-cases 3
```

What it measures today:

- topic coverage rate against expected answer topics
- required uncertainty statement pass rate

## How to read the evidence conservatively

- `retrieval` metrics are the strongest current evidence in the repo.
- `grounding` and `safety` are now explicitly wired, but they are still rule-based automated checks, not clinician review.
- `product_behavior` metrics are useful zero-cost local checks, not clinician safety validation.
- Dry-run reports keep true `safety` metrics as `not_evaluated`; `product_behavior` is reported separately for deterministic local gate coverage only.
- The runner marks clinician review as `not_evaluated` until a human rubric is applied.
- If a reranked mode underperforms the baseline, keep the baseline as the default and document the failure mode.
