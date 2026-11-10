# Run report: Compare geometry ranking on shift_in

- Run ID: 2026-11-10-009/20261110-222601-remaining-shift-in
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:26:01+11:00
- Ended: 2026-11-10T22:26:19+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/shift_in.json (3fcea33dbf38c53376040dc1176c00e0dc583a9009831ecfe16ca1916d226568)
- Seeds: 11
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC plus geometry-ranked pinch
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/shift_in.json --out C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/shift_in`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Measure transfer and possible regression on the remaining established cube fixtures.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"shift_in","pose":{"candidates":6,"offset_m":[0.0,0.0,0.028749999999999998],"rotation":[[1.0,-0.0,0.0],[0.0,0.9659258262890682,0.25881904510252074],[-0.0,-0.25881904510252074,0.9659258262890682]],"score":1.0276142884650206,"support_links":2,"predicted_closure_fraction":0.5},"stop":"grip lost","phases":"grasp_geometry,settle,raise,pregrasp,reach,align_1,align_2,descend,preclose,close,probe,lift","peak":0.10254674394268204,"verified":false}

## Regressions observed

Compare with the prior controller in the task report.

## Interruptions and failures

Execution completed. See Results and the trace for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/shift_in (dir, sha256 c722d79503a56bec2eddb813e6426920d846b41557f61d541c85b7955454e646)
