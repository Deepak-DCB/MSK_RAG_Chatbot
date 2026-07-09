# Concept Graph Completeness Review - 2026-05-21

This second read-only review compared the rebuilt graph against all 20 source articles after Round B/C expansion. The graph now has substantially broader coverage, but it is not yet complete enough to claim full 20-article graph completeness.

## Rebuilt Graph Baseline

- Graph manifest timestamp: `2026-05-21T15:10:14Z` during review.
- Current graph family: deterministic rules, no LLM extraction.
- Current graph counts after Round C rebuild: 225 nodes, 3,671 edges/claims, 22 paths during review; subsequent path-policy tuning may update path count.

## Overall Verdict

The graph has moved from TOS-biased coverage toward broad article-family coverage. Remaining gaps are now mainly quality-control problems:

- some article-specific path families are still missing or too thin;
- some generated edges are anatomically overbroad despite target-type checks;
- some path directions are odd because path construction can traverse edges in either direction;
- several safety-critical neurovascular and medication-adjacent topics require clinician review before stronger language;
- a few aliases still create false positives, especially `axis` matching generic axis wording and `levator` matching non-scapular levator muscles.

## Article Status Table

| Article | Status | Main remaining issue |
| --- | --- | --- |
| Tinnitus, neck and TMJ | unsafe_or_overconfident | Needs cleaner TMJ/eustachian/tympanic/cranial nerve path and stricter false-positive edge guards |
| Atlas joint instability | unsafe_or_overconfident | Needs stabilization/alignment path; avoid unrelated lumbar/pelvic false-positive edges |
| Hip pain | missing_path_family | Needs full psoas/posture -> anterior femoral glide -> FAI/labral -> hip/groin path |
| Chronic muscle clencher | missing_core_nodes | Needs article-local fatigue/tiredness detection and stricter cross-region compression guards |
| AAI/CCI overdiagnosis | needs_clinician_review | Safety paths exist but require clinician review and differential/overdiagnosis framing |
| TOS | missing_path_family | Needs vascular/autonomic TOS path and less leakage into lumbar/pelvic paths |
| Lumbar plexus compression | unsafe_or_overconfident | Needs directionally coherent psoas -> lumbar plexus -> peripheral nerves -> groin/pelvic path; stricter innervation pairs |
| Intracranial hypertension | needs_clinician_review | Good coverage but safety-critical and still has weak same-section bridges |
| Chronic neck pain | missing_path_family | Needs chronic-neck-pain-specific multifactorial/active intervention path |
| ME/CFS biomechanics | missing_core_nodes | Missing or under-detected venous obstruction, ICP/CSF phrasing, orthostatic dysfunction, brain fog detail |
| Pudendal/genital pain | partial | Core paths exist; missing differential concepts such as tumor, MS, diabetes, iatrogenic injury, electrodiagnostic uncertainty |
| Scapular dyskinesis | partial | Needs motor-control-vs-strength policy edge and richer article-specific shoulder control framing |
| POTS/craniovascular | needs_clinician_review | Core paths exist; lacks medication/sodium/vasopressor caution and stroke-like symptom specificity |
| Lumbar lordosis/APT | missing_core_edges | Needs APT myth/differential posture framing, not just posterior pelvic tilt/lordosis path |
| Knee pain/alignment | missing_path_family | Needs valgus/external rotation and anterior tibial glide branches to PFPS/knee pain |
| Shoulder pain/scapular stability | unsafe_or_overconfident | Needs depressed scapular resting position; fix `axis` false positive; avoid overgenerated stabilization edges |
| Migraine/atlas/TOS | unsafe_or_overconfident | Needs migraine-specific craniovascular/venous path; avoid unsupported direct vascular compression claims |
| TMD | missing_path_family | Needs mandibular retraction/TMJ compression/disc displacement/jaw pain path and occlusion/maxilla/mouth-breathing concepts |
| Lower back pain | missing_core_nodes | Needs `nerve root`, plural disc-herniation aliases, and disc-loading -> herniation -> pain path |
| Vestibular/neck/TMJ | missing_path_family | Needs instantiated vestibular-specific path and BPPV/Meniere/cervicogenic dizziness differential concepts |

## High-Priority Completeness Tests To Add Next

- TMJ/tinnitus path must include TMJ/eustachian or tympanic terms, not only mandible/TMD/pterygoid terms.
- `levator veli palatini` must not map to `levator scapulae`.
- `axis of rotation` and `humeral axis` must not map to C2 `axis`.
- Hip path must include psoas or posterior pelvic tilt, anterior femoral glide, FAI/labral impingement, and hip/groin pain.
- Lumbar plexus path must preserve psoas/lumbar region -> lumbar plexus -> peripheral nerves -> groin/hip/pelvic pain direction.
- `innervates` and `supplies` should use approved anatomical source-target pairs, not only target type checks.
- Chronic neck pain needs a non-TOS multifactorial path.
- TOS vascular/autonomic coverage needs a conservative vascular/autonomic path.
- AAI/CCI, ICH, POTS, craniovascular, vestibular, and migraine neurovascular paths need safety review and clinician-review status before completion claims.
- Disc herniation aliases must catch plural forms and lower-back article phrasing; add `nerve root`.
- Vestibular differential concepts such as BPPV, Meniere's disease, cervicogenic dizziness, and vestibular neuritis should be represented without unsupported causal claims.

## Completion Position

The graph should currently be described as: broader deterministic article-family graph with conservative policies and known incompleteness. It should not yet be described as corpus-complete or clinically validated.

## 2026-05-23 Batch 3 Update

The strict 20-case graph completeness probe now passes as an automated coverage check after targeted deterministic path-continuity work. This means each source-article family has at least one reviewable query-to-node/path representation in `datasets/graph-completeness-cases.jsonl`; it does not mean the graph is clinically validated or exhaustive.

- Current rebuilt graph counts: 231 nodes, 4,111 edges/claims, 31 paths.
- The completeness dataset now marks all 20 cases as `strict_probe_passed` with `known_gap: false`.
- `tests/test_graph_completeness_dataset.py` now enforces all 20 cases instead of xfail-marking them.
- Validation: `pytest tests/test_concept_graph_build.py tests/test_graph_retrieval.py tests/test_graph_coverage.py tests/test_graph_completeness_dataset.py` passed with 30 tests.
- Dry-run eval artifacts were written under `Evaluation/runs/20260523_183717_851002_97bb0c7_off/` and `Evaluation/runs/20260523_183729_132155_97bb0c7_off/`.

Remaining caution: safety-critical neurovascular, vestibular, craniovascular, AAI/CCI, ICH, POTS, and medication-adjacent paths still require conservative policy labels and clinician review before stronger product claims. The graph may be described as passing the current deterministic 20-article coverage probe, not as clinically validated or corpus-exhaustive.
