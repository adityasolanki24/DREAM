# Run report: R1 fixed loaded wrist orientation on sphere

- Run ID: 2026-11-10-010/20261110-223947-wrist-sphere
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:39:47+11:00
- Ended: 2026-11-10T22:40:01+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/sphere.json (bd00fda8e33e6a86f8ed15118c175e116ad14d7b80775e35fed24a6a38748047)
- Seeds: 43
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/sphere.json --out C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/sphere`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check whether preserving the initial loaded wrist orientation improves the static-ranking execution.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold failed. Peak lift was -0.0002 m.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The live controller stopped for grip lost.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/sphere (dir, sha256 c38f9501a464735d356af88a5b16036da9a73c9bfac56aff534d1a8589d2a60c)
