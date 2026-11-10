# Run report: R1 physical rehearsal and live replay of fresh_far

- Run ID: 2026-11-10-010/20261110-223328-rehearsal-fresh-far
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:33:28+11:00
- Ended: 2026-11-10T22:33:43+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/fresh_far.json (1e9b903ec891ca9f303ffc1f53611cce5dcaa1cbb05bbf12a6d306a64c78e798)
- Seeds: 31
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/fresh_far.json --out C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/fresh_far`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Screen the top geometry candidates with a prescribed wrist, then replay the selection on the live free-standing G1.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold passed. Peak lift was 0.1367 m. 0 of 12 predicted holds cleared the height gate. The live run used rank 0.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The run completed without interruption.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/fresh_far (dir, sha256 6fe79976ecfe0775a85fadb0e364429f5a85eec9d157f75db3863f362f16483a)
