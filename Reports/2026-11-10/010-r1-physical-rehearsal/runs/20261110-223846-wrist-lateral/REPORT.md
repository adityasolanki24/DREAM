# Run report: R1 fixed loaded wrist orientation on lateral

- Run ID: 2026-11-10-010/20261110-223846-wrist-lateral
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:38:46+11:00
- Ended: 2026-11-10T22:39:02+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/lateral.json (b377c36078c97ae0fcbab7e1f2f85005278b546e57f5bdf4420279f032d8b571)
- Seeds: 23
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/lateral.json --out C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/lateral`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check whether preserving the initial loaded wrist orientation improves the static-ranking execution.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold passed. Peak lift was 0.1367 m.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The run completed without interruption.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/lateral (dir, sha256 fd7c9e199eccbd5d0a8d69f3444c301b6ece51f43a11ab39946405a989681c10)
