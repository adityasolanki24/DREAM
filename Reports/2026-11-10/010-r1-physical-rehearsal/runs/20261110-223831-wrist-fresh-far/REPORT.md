# Run report: R1 fixed loaded wrist orientation on fresh_far

- Run ID: 2026-11-10-010/20261110-223831-wrist-fresh-far
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:38:31+11:00
- Ended: 2026-11-10T22:38:45+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/fresh_far.json (cad148f05ba5df75a03476ad24c055bd1769f607fdeeb493a5185efcd71f5690)
- Seeds: 31
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/fresh_far.json --out C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/fresh_far`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check whether preserving the initial loaded wrist orientation improves the static-ranking execution.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold passed. Peak lift was 0.1344 m.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The run completed without interruption.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/fresh_far (dir, sha256 8f1fd1ad7336fc65d3ea1ae868af4ceeb253f7c66e22069eb3a99973fcd3c652)
