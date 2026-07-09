# Concept Graph Completion Spec - Round B

This spec defines the current non-TOS MSK completion target for the deterministic concept graph. It does not claim clinician validation or diagnostic authority. Completion means the graph can retrieve article-relevant concepts, conservative edges, and reviewable mechanism paths for each listed domain without known false-positive aliases.

## Round B Scope

Round B prioritizes non-TOS MSK article families before safety-critical neurovascular expansion.

| Domain | Required concepts | Required path target | Forbidden / guarded behavior |
| --- | --- | --- | --- |
| Hip pain | hip joint, femoroacetabular impingement, anterior femoral glide, labral impingement, hip/groin pain, psoas/gluteal concepts | posture or psoas/femoral glide -> FAI/labral or hip joint load -> hip/groin pain | Do not satisfy hip queries with TOS-only paths |
| Knee alignment | knee pain, tibia, patella, posterior/anterior tibial glide, functional varus/valgus, patellofemoral pain, jumper's knee | pelvic or tibial alignment -> tibial glide/rotation/patellar tracking -> knee pain pattern | Do not require TOS/scapular concepts for knee coverage |
| Lumbar/lordosis/back pain | posterior/anterior pelvic tilt, lumbar lordosis, lumbar spine, disc loading, disc herniation, low back pain | pelvic/lumbar posture -> disc loading/lordosis -> disc herniation or low back pain | Anterior pelvic tilt must not map to anterior scapular tilt |
| Lumbar plexus | psoas, lumbar plexus, femoral/genitofemoral/ilioinguinal/lateral femoral cutaneous nerves, groin/hip/pelvic pain | psoas/lumbar region -> lumbar plexus/peripheral nerves -> groin/hip/pelvic pain | Lumbar plexus must not map to brachial plexus |
| Pudendal/hypogastric pelvic pain | pudendal nerve, pudendal neuralgia, inferior hypogastric plexus, pelvic floor, Alcock's canal, piriformis/obturator internus, pelvic pain | pelvic floor/deep rotator/Alcock's canal -> pudendal or hypogastric plexus -> pelvic/genital pain | Pelvic/hypogastric plexus must not map to brachial plexus |
| Shoulder/scapular | scapular dyskinesis, scapular winging, subacromial space, rotator cuff, supraspinatus/subscapularis, shoulder impingement, shoulder pain | scapular dyskinesis/control -> subacromial or glenohumeral mechanics -> rotator cuff/impingement -> shoulder pain | Do not collapse weak shoulder paths into diagnosis |
| Clenching/bracing | protective bracing, chronic muscle clenching/GICS, manual muscle testing, altered breathing mechanics, dyspnea, fatigue, neuralgia/neck pain | clenching/bracing -> MMT or breathing distortion -> fatigue/persistent symptoms | Do not represent bracing as a definitive diagnosis |

## Acceptance Checks

- Coverage cases in `datasets/graph-coverage-cases.jsonl` must find expected nodes and avoid forbidden nodes.
- Each Round B domain should return at least one article-relevant path or a documented exclusion.
- Known alias collisions remain blocked by tests.
- Edges with `supplies` or `innervates` cannot target symptoms, conditions, or red flags.
- Paths with weak or inferred edges must retain conservative policy labels.

## Remaining Work

Round C will cover safety-critical neurovascular graph completion: AAI/CCI, atlas instability, intracranial hypertension, POTS/craniovascular, migraine/atlas/TOS, tinnitus/TMJ, and vestibular mechanisms.
