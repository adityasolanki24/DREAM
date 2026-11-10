# Run report: Test the same planner on a rectangular

- Run ID: 2026-11-10-009/20261110-222428-shape-rectangular
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:24:28+11:00
- Ended: 2026-11-10T22:24:42+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-shapes-contact-001/rectangular.json (6e4b0c1f67d78c004e4f292396e9a4f3fba2c38c45058e41c085dd83d39ec187)
- Seeds: 37
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC plus shared geometry-ranked pinch
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-shapes-contact-001/rectangular.json --out C:/Users/24adi/DREAM-r1-evidence/r1-shapes-contact-001/rectangular`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Exercise the same candidate search and feedback on new observed object geometry.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"rectangular","pose":{"candidates":15,"offset_m":[0.0,-0.005000000000000001,0.0325],"rotation":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"score":1.130570375965831,"support_links":2,"predicted_closure_fraction":0.375},"stop":"grip lost","phases":"grasp_geometry,settle,raise,pregrasp,reach,align_1,descend,preclose,close,probe","peak":0.0020334839638417312,"verified":false}

## Regressions observed

This is a new development fixture with no prior result.

## Interruptions and failures

Execution completed. See Results and the trace for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-shapes-contact-001/rectangular (dir, sha256 56fdfe6ddb250314d1e848a32ccf83e9c0702454b866be2b90bc75382d0456fe)
