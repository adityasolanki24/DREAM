# Run report: R1 physical rehearsal and live replay of sphere

- Run ID: 2026-11-10-010/20261110-223408-rehearsal-sphere
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:34:08+11:00
- Ended: 2026-11-10T22:34:20+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/sphere.json (171986741c82d2e64119068f73dcdfd798d6eda41d54af5899089ac1b2ce77b7)
- Seeds: 43
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/sphere.json --out C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/sphere`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Screen the top geometry candidates with a prescribed wrist, then replay the selection on the live free-standing G1.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold failed. Peak lift was -0.0002 m. 0 of 12 predicted holds cleared the height gate. The live run used rank 0.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The live controller stopped for grip lost.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/sphere (dir, sha256 06748ddcc04465ee4000653103c61b15288578145ba90832733444d9207bb7ee)
