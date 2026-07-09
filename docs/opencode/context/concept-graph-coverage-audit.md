# Concept Graph Coverage Audit - 2026-05-20

This read-only audit compared the deterministic concept graph against the 20 hierarchical source articles. The graph is useful for TOS, scalene, scapular, and neck-related mechanism context, but it is not a complete representation of the corpus.

## Current Baseline

- Source: `MSKArticlesINDEX/hierarchical/` with 20 articles and 2,324 evidence spans.
- Graph artifacts: `MSKArticlesINDEX/graph/`.
- Baseline graph counts: 77 nodes, 1,113 edges, 1,113 claims, 4 paths.
- Baseline support mix: 667 weak edges, 369 indirect edges, 71 direct edges, 6 inferred same-section edges.

## Corpus Coverage Verdict

The graph is TOS-biased and alias-dependent. It misses major article families including hip, knee, lumbar, pelvic/genital, TMJ, tinnitus, vestibular, POTS, ME/CFS, atlas instability, intracranial hypertension, and broader chronic neck pain models.

| Article | Coverage verdict | Main gaps |
| --- | --- | --- |
| Tinnitus, neck and TMJ | Not complete | Tinnitus, TMJ, ear anatomy, cranial nerves, auditory and vestibular pathways |
| Atlas joint instability | Not complete | Atlas/C1/C2, A-A/A-O joints, ligament laxity, suboccipitals, stabilization model |
| Hip pain | Near-zero | Hip joint, FAI, bursitis, psoas/gluteal mechanisms, femoral glide paths |
| Chronic muscle clencher | Not complete | Clenching/GICS/bracing, fatigue, breath holding, manual muscle testing distortion |
| AAI/CCI overdiagnosis | Not complete and safety-thin | AAI, CCI, brainstem, jugular/vertebral compromise, imaging/differential framework |
| TOS | Best-covered but incomplete | Diagnostic controversy, TOCS nuance, vascular/autonomic details |
| Lumbar plexus compression | Not complete | Lumbar plexus, psoas entrapment, groin/pelvic referral |
| Intracranial hypertension | Not complete | ICH/IIH, CSF pressure/leak, venous sinus stenosis, papilledema, venous drainage |
| Chronic neck pain | Partial | Multifactorial neck-pain framing, broader intervention guidance |
| ME/CFS biomechanical | Not complete | ME/CFS, fatigue, orthostatic dysfunction, venous obstruction, intracranial pressure |
| Pudendal/genital pain | Very incomplete | Pudendal nerve, inferior hypogastric plexus, pelvic floor, pelvic entrapment pathways |
| Scapular dyskinesis | Partial | Rotator cuff, scapular winging, shoulder impingement mechanics |
| POTS/craniovascular | Partial | POTS, orthostatic symptoms, venous/jugular/hemodynamic treatment cautions |
| Lumbar lordosis/APT | Very incomplete | Pelvic tilt, lumbar lordosis, disc loading, swayback differential |
| Knee pain/alignment | Essentially absent | Knee anatomy, patellar tracking, tibial glide, varus/valgus, tissue-load paths |
| Shoulder pain/scapular stability | Partial but shallow | Rotator cuff, humeral head, glenoid/acromion/coracoid impingement paths |
| Migraine/atlas/TOS | Moderate for TOS, weak for atlas/venous migraine | IJV, atlas/C1/C2, venous congestion, vertebral artery/vein pathways |
| TMD | Very low | TMJ, mandible, pterygoids, occlusion, disc displacement, jaw mechanics |
| Lower back pain | Nearly absent | Lumbar spine, posterior pelvic tilt, disc herniation, nerve root, extensor inhibition |
| Vestibular/neck/TMJ | Low-to-moderate | Vestibular/TMJ/ear-pressure mechanisms and cranial nerves |

## High-Priority Correctness Issues

- The alias `plexus` was too broad for `brachial plexus` and could incorrectly match lumbar, sacral, pudendal, or hypogastric plexus passages.
- The alias `anterior tilt` was too broad for `anterior scapular tilt` and could incorrectly match anterior pelvic tilt passages.
- Vascular `supplies` and nerve `innervates` rules could create semantically invalid edges to symptoms or conditions when source text used supply/innervation wording near non-anatomical concepts.

## Acceptance Standard

Do not describe the graph as corpus-complete until each article has reviewed coverage for nodes, aliases, mechanism edges, safety/differential claims, tests/interventions, and path families. Until then, describe it as a deterministic, evidence-linked mechanism graph with uneven article coverage.
