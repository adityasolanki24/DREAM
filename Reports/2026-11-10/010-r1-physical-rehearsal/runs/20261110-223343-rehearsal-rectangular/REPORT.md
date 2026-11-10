# Run report: R1 physical rehearsal and live replay of rectangular

- Run ID: 2026-11-10-010/20261110-223343-rehearsal-rectangular
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:33:43+11:00
- Ended: 2026-11-10T22:33:55+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/rectangular.json (e3d6328e48c5afd7ceebb53d7da15278360d213ffc91a8434618b0c4fcc6194c)
- Seeds: 37
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/rectangular.json --out C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/rectangular`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Screen the top geometry candidates with a prescribed wrist, then replay the selection on the live free-standing G1.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold failed. Peak lift was 0.0020 m. 0 of 12 predicted holds cleared the height gate. The live run used rank 0.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The live controller stopped for grip lost.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/rectangular (dir, sha256 2a9f2f7633416987ce01c05569f60dadb4fa43559cb7bd80f019cfefab370817)
