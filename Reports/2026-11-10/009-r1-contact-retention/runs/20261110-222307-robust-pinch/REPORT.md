# Run report: Evaluate uncertainty-screened pinch on pinch

- Run ID: 2026-11-10-009/20261110-222307-robust-pinch
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:23:07+11:00
- Ended: 2026-11-10T22:23:30+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-robust-contact-001/pinch.json (cc30700cd3142df6da0d0770dea52b6e6b5d6dcf1043da25e4fe95d32c4546e0)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC plus uncertainty-screened pinch
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-robust-contact-001/pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-robust-contact-001/pinch`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test whether selecting a pose robust to small position errors improves physical retention.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"pinch","pose":{"candidates":8,"offset_m":[0.0,0.0,0.025],"rotation":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"score":0.9895212752361494,"support_links":2,"predicted_closure_fraction":1.0},"stop":null,"phases":"grasp_geometry,settle,raise,pregrasp,reach,align_1,align_2,descend,preclose,close,probe,lift","peak":0.15063755805608414,"verified":true}

## Regressions observed

Compare with the geometry-only and nominal controller runs in this task.

## Interruptions and failures

Execution completed. See the recorded result for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-robust-contact-001/pinch (dir, sha256 a6c9d15c30ab8992085556838eb395ae381b9bfb4b3268ae1c9a1fb2f1579960)
