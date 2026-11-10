# Run report: Recheck shift_out on the shape-generalized selector

- Run ID: 2026-11-10-009/20261110-222744-final-shift-out
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:27:44+11:00
- Ended: 2026-11-10T22:27:54+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/shift_out.json (468f01833d8434c7741ee4ab647332a4a174aa979592ec641955c81c2e4f11b5)
- Seeds: 17
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC plus geometry-ranked pinch
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/shift_out.json --out C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/shift_out`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Verify the final shared geometry selector on the matched cube fixture.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"shift_out","stop":"grip lost","peak":0.002219579811586936,"verified":false}

## Regressions observed

Matched comparison recorded in the aggregate evidence.

## Interruptions and failures

Execution completed. The physical outcome is recorded in Results.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/shift_out (dir, sha256 e797e1859cb1c0dcab502f40ae2969ba1c62fe789ecd61456346d8e24823e64f)
