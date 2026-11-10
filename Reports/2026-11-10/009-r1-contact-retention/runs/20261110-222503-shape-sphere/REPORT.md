# Run report: Test the same planner on a sphere

- Run ID: 2026-11-10-009/20261110-222503-shape-sphere
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:25:03+11:00
- Ended: 2026-11-10T22:25:20+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-shapes-contact-001/sphere.json (7765331ea21723f3cf865a23638312077868948214294e28a2c429ecd078ac14)
- Seeds: 43
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC plus shared geometry-ranked pinch
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-shapes-contact-001/sphere.json --out C:/Users/24adi/DREAM-r1-evidence/r1-shapes-contact-001/sphere`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Exercise the same candidate search and feedback on new observed object geometry.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"sphere","pose":{"candidates":15,"offset_m":[0.0,0.0,0.025],"rotation":[[1.0,-0.0,0.0],[0.0,0.9659258262890682,0.25881904510252074],[-0.0,-0.25881904510252074,0.9659258262890682]],"score":1.0660618014006067,"support_links":2,"predicted_closure_fraction":0.625},"stop":"grip lost","phases":"grasp_geometry,settle,raise,pregrasp,reach,align_1,descend,preclose,close,probe","peak":-0.0002458796004892738,"verified":false}

## Regressions observed

This is a new development fixture with no prior result.

## Interruptions and failures

Execution completed. See Results and the trace for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-shapes-contact-001/sphere (dir, sha256 95278a7a42a5f6b7f6ad1072bed0412f732fadb7d71943570cf3999046a7bf17)
