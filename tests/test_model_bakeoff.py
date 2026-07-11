"""Zero-cost tests for the model bake-off's aggregation + ranking logic."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import model_bakeoff as MB


def _row(faith, mode="llm:groq", tok=100, gen=1.0):
    return {"q": "q", "faithfulness": faith, "n_claims": 3,
            "answer_mode": mode, "output_tokens": tok, "gen_time": gen}


def test_aggregate_basic_means():
    rows = [_row(1.0, tok=100, gen=2.0), _row(0.5, tok=200, gen=4.0)]
    a = MB.aggregate(rows)
    assert a["n"] == 2 and a["n_scored"] == 2
    assert a["faithfulness"] == 0.75
    assert a["avg_tokens"] == 150
    assert a["avg_gen_s"] == 3.0
    assert a["llm_rate"] == 1.0
    assert a["refusal_rate"] == 0.0


def test_aggregate_counts_evidence_only_and_none_as_refusals():
    rows = [
        _row(1.0, mode="llm:groq"),
        _row(None, mode="evidence_only"),   # degraded fallback
        _row(None, mode="llm:groq"),        # produced text but 0 claims (refusal)
    ]
    a = MB.aggregate(rows)
    # refusals: the evidence_only + the no-claim answer = 2/3
    assert abs(a["refusal_rate"] - 2 / 3) < 1e-9
    assert abs(a["llm_rate"] - 2 / 3) < 1e-9   # 2 of 3 answer_mode start with llm:
    assert a["n_scored"] == 1                  # only one had a numeric score
    assert a["faithfulness"] == 1.0


def test_aggregate_empty_rows():
    a = MB.aggregate([])
    assert a["n"] == 0 and a["faithfulness"] is None
    assert a["refusal_rate"] == 0.0 and a["avg_tokens"] == 0.0


def test_format_table_ranks_by_faithfulness_desc():
    results = {
        "low": {"candidate": {"name": "low"}, "agg": MB.aggregate([_row(0.2)])},
        "high": {"candidate": {"name": "high"}, "agg": MB.aggregate([_row(0.9)])},
    }
    table = MB.format_table(results)
    # 'high' must appear before 'low' in the ranked table body.
    assert table.index("high") < table.index("low")
