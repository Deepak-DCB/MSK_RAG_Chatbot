"""Smoke tests for the eval CLI's argparse defaults.

Regression coverage for a bug where DEFAULT_DATASET pointed at a gitignored
path (Eval/gold_set_v2.jsonl) that never existed in a clean checkout, so the
documented no-flag invocation — and CI's smoke step — failed at startup with
"Dataset not found" for weeks. Unit tests over the scoring functions never
caught it because nothing exercised main()/argparse end to end.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT = PROJECT_ROOT / "scripts" / "run_eval_production.py"


def test_default_dataset_exists_in_checkout():
    from scripts.run_eval_production import DEFAULT_DATASET

    assert DEFAULT_DATASET.exists(), (
        f"DEFAULT_DATASET points at {DEFAULT_DATASET}, which is not committed; "
        "the no-flag CLI invocation documented in docs/evaluation.md would fail"
    )


def test_dry_run_with_no_dataset_flag_exits_zero():
    # Writes a run under the gitignored Evaluation/runs/, same as CI's smoke step.
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--max-cases",
            "2",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
