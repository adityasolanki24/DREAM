# Run report: Evaluate geometry-ranked pinch on shift_out

- Run ID: 2026-11-10-009/20261110-222009-geometry-shift-out
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:20:09+11:00
- Ended: 2026-11-10T22:20:32+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-contact-search-001/shift_out.json (468f01833d8434c7741ee4ab647332a4a174aa979592ec641955c81c2e4f11b5)
- Seeds: 17
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC plus geometry-ranked pinch
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-contact-search-001/shift_out.json --out C:/Users/24adi/DREAM-r1-evidence/r1-contact-search-001/shift_out`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Compare a geometry-ranked pinch against the prior hold on the same scene.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"shift_out","pose":{"candidates":30,"offset_m":[0.0,-0.005600000000000001,0.035],"rotation":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],"score":1.1172097067316582,"support_links":2,"predicted_closure_fraction":0.375},"stop":"grip lost","phases":"grasp_geometry,settle,raise,pregrasp,reach,align_1,descend,preclose,close,probe","peak":0.002219579811586936,"verified":false}

## Regressions observed

See the recorded hold outcome and the comparison in the task report.

## Interruptions and failures

Execution completed. See Results and the trace for the physical result.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-contact-search-001/shift_out (dir, sha256 e797e1859cb1c0dcab502f40ae2969ba1c62fe789ecd61456346d8e24823e64f)
