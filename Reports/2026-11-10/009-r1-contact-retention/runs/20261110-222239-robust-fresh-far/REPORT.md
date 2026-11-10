# Run report: Evaluate uncertainty-screened pinch on fresh_far

- Run ID: 2026-11-10-009/20261110-222239-robust-fresh-far
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:22:39+11:00
- Ended: 2026-11-10T22:23:06+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-robust-contact-001/fresh_far.json (4c3c24044ea38303db0b1d2090a4e75365fe6710b58b195e41b1ad8ae3a5ddee)
- Seeds: 31
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC plus uncertainty-screened pinch
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-robust-contact-001/fresh_far.json --out C:/Users/24adi/DREAM-r1-evidence/r1-robust-contact-001/fresh_far`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test whether selecting a pose robust to small position errors improves physical retention.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"fresh_far","pose":{"candidates":8,"offset_m":[0.0,0.0,0.025],"rotation":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"score":0.9895212752361499,"support_links":2,"predicted_closure_fraction":1.0},"stop":"grip lost","phases":"grasp_geometry,settle,raise,pregrasp,reach,align_1,align_2,descend,preclose,close,probe,lift","peak":0.14072232178522381,"verified":false}

## Regressions observed

Compare with the geometry-only and nominal controller runs in this task.

## Interruptions and failures

Execution completed. See the recorded result for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-robust-contact-001/fresh_far (dir, sha256 7367f7b39ef96779f3d03bfc96e3fd75bdf3f6a054150220afe6d03d19f29223)
