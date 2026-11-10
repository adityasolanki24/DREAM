# Run report: R1 actuated rehearsal and live replay of pinch

- Run ID: 2026-11-10-010/20261110-223529-stage-pinch
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:35:29+11:00
- Ended: 2026-11-10T22:35:41+11:00
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
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-stage-001/pinch`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Recheck candidate retention with a force-limited vertical stage and the same slip-to-preload regulator as the live G1.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold failed. Peak lift was 0.0308 m. 2 of 12 predicted holds cleared the height gate. The live run used rank 4.

## Regressions observed

This variant lost a previously verified hold.

## Interruptions and failures

The live controller stopped for grip lost.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-stage-001/pinch (dir, sha256 63d409336b35892e00e64a3b3b41344e499719733eec7818a5459463ac8c9645)
