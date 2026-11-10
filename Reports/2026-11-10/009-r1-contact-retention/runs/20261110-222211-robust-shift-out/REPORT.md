# Run report: Evaluate uncertainty-screened pinch on shift_out

- Run ID: 2026-11-10-009/20261110-222211-robust-shift-out
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:22:11+11:00
- Ended: 2026-11-10T22:22:37+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-robust-contact-001/shift_out.json (e8c94991d4583f5dcb4e197338f5b8da3039b5dc991aa501f093b1fa368169aa)
- Seeds: 17
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC plus uncertainty-screened pinch
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-robust-contact-001/shift_out.json --out C:/Users/24adi/DREAM-r1-evidence/r1-robust-contact-001/shift_out`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test whether selecting a pose robust to small position errors improves physical retention.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"shift_out","pose":{"candidates":7,"offset_m":[0.0,-0.005600000000000001,0.028],"rotation":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"score":1.0260553901868832,"support_links":2,"predicted_closure_fraction":0.375},"stop":"grip lost","phases":"grasp_geometry,settle,raise,pregrasp,reach,align_1,align_2,descend,preclose,close,probe,lift","peak":0.1424213970008168,"verified":false}

## Regressions observed

Compare with the geometry-only and nominal controller runs in this task.

## Interruptions and failures

Execution completed. See the recorded result for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-robust-contact-001/shift_out (dir, sha256 5955ba317378ea10f4631a87b2f5ce6fbdd65a777ebe28161442fb05365dbc9c)
