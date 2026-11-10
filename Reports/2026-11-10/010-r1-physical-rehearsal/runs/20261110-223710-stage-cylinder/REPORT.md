# Run report: R1 actuated rehearsal and live replay of cylinder

- Run ID: 2026-11-10-010/20261110-223710-stage-cylinder
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:37:10+11:00
- Ended: 2026-11-10T22:37:25+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/cylinder.json (04d07a38ee0519efbdc968ff0288270eab17b7635bd3e8aeba6cf941bcc3cf80)
- Seeds: 41
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/cylinder.json --out C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-stage-001/cylinder`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Recheck candidate retention with a force-limited vertical stage and the same slip-to-preload regulator as the live G1.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold failed. Peak lift was -0.0001 m. 1 of 12 predicted holds cleared the height gate. The live run used rank 9.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The run completed without interruption.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-stage-001/cylinder (dir, sha256 834cf0b2ce7e88777d5daef87bca91a7a610c37bf1bb4b352dc981d4b96735f1)
