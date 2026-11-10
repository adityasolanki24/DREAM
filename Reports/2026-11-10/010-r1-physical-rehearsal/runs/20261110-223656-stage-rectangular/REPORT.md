# Run report: R1 actuated rehearsal and live replay of rectangular

- Run ID: 2026-11-10-010/20261110-223656-stage-rectangular
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:36:56+11:00
- Ended: 2026-11-10T22:37:10+11:00
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
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/rectangular.json --out C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-stage-001/rectangular`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Recheck candidate retention with a force-limited vertical stage and the same slip-to-preload regulator as the live G1.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold passed. Peak lift was 0.1447 m. 2 of 12 predicted holds cleared the height gate. The live run used rank 10.

## Regressions observed

The selector recovered a previous failure.

## Interruptions and failures

The run completed without interruption.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-stage-001/rectangular (dir, sha256 d4d9c44561b7cf2123f1be953224ea62f6a0b0cf303c1b6ac51b1dac7f440542)
