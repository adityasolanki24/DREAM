# Run report: R1 fixed loaded wrist orientation on shift_in

- Run ID: 2026-11-10-010/20261110-223753-wrist-shift-in
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:37:53+11:00
- Ended: 2026-11-10T22:38:06+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/shift_in.json (bb576fb28d1df1954351dc2883cc4d9f103a27ff237c1fdc3aa301a306d9eb47)
- Seeds: 11
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/shift_in.json --out C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/shift_in`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check whether preserving the initial loaded wrist orientation improves the static-ranking execution.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold failed. Peak lift was 0.0988 m.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The live controller stopped for grip lost.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/shift_in (dir, sha256 7b79c65e06aa70faa02a81dabe931b6a2c464b2136f943ec6ab0b5611e1c77ef)
