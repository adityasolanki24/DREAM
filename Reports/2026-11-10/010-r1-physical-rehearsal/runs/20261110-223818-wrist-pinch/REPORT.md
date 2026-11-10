# Run report: R1 fixed loaded wrist orientation on pinch

- Run ID: 2026-11-10-010/20261110-223818-wrist-pinch
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:38:18+11:00
- Ended: 2026-11-10T22:38:31+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/pinch.json (11e8ab897b178159af11972500a7835f2ac250c70abcc5121e239fb8eb3a6a71)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/pinch`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check whether preserving the initial loaded wrist orientation improves the static-ranking execution.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold passed. Peak lift was 0.1422 m.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The run completed without interruption.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/pinch (dir, sha256 fded60f39cc082427e9d7245ff7406956f749b9325fb1dda99398b8e7a80bec5)
