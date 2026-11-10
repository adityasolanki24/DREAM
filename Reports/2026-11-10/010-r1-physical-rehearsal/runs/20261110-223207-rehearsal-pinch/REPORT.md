# Run report: R1 physical rehearsal and live replay of pinch

- Run ID: 2026-11-10-010/20261110-223207-rehearsal-pinch
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:32:07+11:00
- Ended: 2026-11-10T22:32:22+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/pinch.json (bafee083d9af2191dbcef4c70cea36140310e5467a8e3e0cf38678c8b28190bc)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/pinch`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Screen the top geometry candidates with a prescribed wrist, then replay the selection on the live free-standing G1.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold passed. Peak lift was 0.1422 m. 0 of 12 predicted holds cleared the height gate. The live run used rank 0.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The run completed without interruption.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/pinch (dir, sha256 8f3222156b79b61e6e3435d8510da1e5e596e49b49631d1d3143f3c5b353ef84)
