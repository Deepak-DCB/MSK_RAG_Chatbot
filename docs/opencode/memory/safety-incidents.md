# Safety Incident Memory

Track safety-critical misses and escalation failures here.

## Entry template
- Date:
- Case ID:
- Trigger pattern:
- Failure type:
- Severity:
- Immediate corrective action:
- Long-term prevention action:
- Owner and due date:

## 2026-07-18 — negation-scope bypass via contrastive-conjunction hedging
- Date: 2026-07-18
- Case ID: redflag-051 … redflag-054 (datasets/red-flag-cases.jsonl)
- Trigger pattern: a discourse hedge containing a negation word before the
  symptom clause — "I don't want to worry/alarm you, but <red-flag symptom>",
  "Not to be dramatic, but <red-flag symptom>". The negation lookback in
  `_is_negated` treated ANY negation word within 70 chars as negating the
  matched phrase, regardless of the ", but" pivot in between. Four verified
  false negatives on `progressive_neurologic_deficit` phrasings; found in code
  review, not from a user report.
- Failure type: false negative / missed escalation (deterministic red-flag gate).
- Severity: high — safety-critical gate; the codebase's own design principle is
  "a false positive costs one message; a false negative misses an emergency".
- Immediate corrective action: `_CLAUSE_BREAK_RE` in VectorDB/qaEngine.py — a
  contrastive conjunction (but/however/although/though/yet/except) between the
  negation cue and the symptom ends the negation's scope, the same way the
  lookback's character class already ends it at a sentence boundary. The
  existing "not need urgent care for" carve-out is retained (different failure
  shape: user pre-emptively dismissing a symptom, zero gap, no pivot).
- Long-term prevention action: hedge phrasings added to the gold set
  (redflag-051…054) and pinned in tests/test_review_regressions.py, alongside
  true-negative guards. Residual known gap (pre-existing, direction unchanged
  by this fix): `_is_negated` only inspects text BEFORE the match start, so a
  negation inside a multi-word pattern's own gap (e.g. "weakness has not been
  getting worse") is invisible to it, and `re.finditer`'s non-overlapping scan
  can let an early negated alternative swallow a later unnegated one. Follow-up
  candidate for the next safety review round.
- Owner and due date: found/fixed in bug-fixing round of 2026-07-18; merge
  requires explicit human sign-off (protected surface per
  docs/opencode/rules/25-bounded-autonomy.md).
