# Run report: Evaluate geometry-ranked pinch on pinch

- Run ID: 2026-11-10-009/20261110-222059-geometry-pinch
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:20:59+11:00
- Ended: 2026-11-10T22:21:17+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-contact-search-001/pinch.json (4f65f639a9b83212df34bff6fb78b28cd00b3fc04d0e27b350ec29db6e7d1e4b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC plus geometry-ranked pinch
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-contact-search-001/pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-contact-search-001/pinch`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Compare a geometry-ranked pinch against the prior hold on the same scene.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"pinch","pose":{"candidates":15,"offset_m":[0.0,0.0,0.025],"rotation":[[1.0,-0.0,0.0],[0.0,0.9659258262890682,0.25881904510252074],[-0.0,-0.25881904510252074,0.9659258262890682]],"score":1.1007298714102947,"support_links":2,"predicted_closure_fraction":0.4375},"stop":null,"phases":"grasp_geometry,settle,raise,pregrasp,reach,align_1,descend,preclose,close,probe,lift","peak":0.14223391185328682,"verified":true}

## Regressions observed

See the recorded hold outcome and the comparison in the task report.

## Interruptions and failures

Execution completed. See Results and the trace for the physical result.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-contact-search-001/pinch (dir, sha256 6c79d41f3ec118cdd447979085b86e67fbd1fff1c49705abb2e9b3ae601f0236)
