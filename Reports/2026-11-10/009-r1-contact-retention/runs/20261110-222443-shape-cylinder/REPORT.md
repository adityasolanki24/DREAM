# Run report: Test the same planner on a cylinder

- Run ID: 2026-11-10-009/20261110-222443-shape-cylinder
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:24:43+11:00
- Ended: 2026-11-10T22:25:01+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-shapes-contact-001/cylinder.json (ecb6de06dd040e4a769537eaa933a0462ccfe70d962cf308c713a20fb345dcd8)
- Seeds: 41
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC plus shared geometry-ranked pinch
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-shapes-contact-001/cylinder.json --out C:/Users/24adi/DREAM-r1-evidence/r1-shapes-contact-001/cylinder`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Exercise the same candidate search and feedback on new observed object geometry.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"cylinder","pose":{"candidates":24,"offset_m":[0.0,0.0,0.03375],"rotation":[[1.0,-0.0,0.0],[0.0,0.9659258262890682,0.25881904510252074],[-0.0,-0.25881904510252074,0.9659258262890682]],"score":1.136816045264665,"support_links":2,"predicted_closure_fraction":0.5625},"stop":"grip lost","phases":"grasp_geometry,settle,raise,pregrasp,reach,align_1,align_2,descend,preclose,close,probe,lift","peak":0.05073144876173874,"verified":false}

## Regressions observed

This is a new development fixture with no prior result.

## Interruptions and failures

Execution completed. See Results and the trace for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-shapes-contact-001/cylinder (dir, sha256 c1e6631fa4c85a07ac7b3a591e1efd02a131a67ef7e6df19422c36f963da4910)
