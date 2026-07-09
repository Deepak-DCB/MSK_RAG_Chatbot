# Instruction Template: Retrieval Tuning

Use this template for retrieval/reranker changes.

## Required process
1. Record baseline metrics and config.
2. Change one tuning lever at a time.
3. Run dry-run validation first.
4. Run bounded evaluation and compare deltas.
5. Keep only changes with net positive quality and no safety regression.

## Required output
- Baseline vs candidate metrics
- Latency and cost impact
- Known tradeoffs
- Recommendation: keep, iterate, or rollback
