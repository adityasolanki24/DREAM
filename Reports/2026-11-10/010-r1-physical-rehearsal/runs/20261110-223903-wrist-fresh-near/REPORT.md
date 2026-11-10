# Run report: R1 fixed loaded wrist orientation on fresh_near

- Run ID: 2026-11-10-010/20261110-223903-wrist-fresh-near
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:39:03+11:00
- Ended: 2026-11-10T22:39:17+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/fresh_near.json (ef975c81ec06e8f2cf248dd68635218cb550bfafde7e97d51376674d2e78ee0e)
- Seeds: 29
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/fresh_near.json --out C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/fresh_near`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check whether preserving the initial loaded wrist orientation improves the static-ranking execution.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold passed. Peak lift was 0.1358 m.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The run completed without interruption.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/fresh_near (dir, sha256 d51dd598418f0988dcbc4c63b409fbf0c1614e8ea2111078cf4d8d462809b987)
