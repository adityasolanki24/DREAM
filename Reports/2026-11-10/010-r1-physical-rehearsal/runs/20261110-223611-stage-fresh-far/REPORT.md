# Run report: R1 actuated rehearsal and live replay of fresh_far

- Run ID: 2026-11-10-010/20261110-223611-stage-fresh-far
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:36:11+11:00
- Ended: 2026-11-10T22:36:26+11:00
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
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/fresh_far.json --out C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-stage-001/fresh_far`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Recheck candidate retention with a force-limited vertical stage and the same slip-to-preload regulator as the live G1.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold failed. Peak lift was 0.0412 m. 2 of 12 predicted holds cleared the height gate. The live run used rank 4.

## Regressions observed

This variant lost a previously verified hold.

## Interruptions and failures

The live controller stopped for contact force.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-stage-001/fresh_far (dir, sha256 06a8d611669152ed0fff85a6da03f389be40acca192b1b9db19579cda874c79c)
