# Run report: R1 actuated rehearsal and live replay of shift_out

- Run ID: 2026-11-10-010/20261110-223542-stage-shift-out
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:35:42+11:00
- Ended: 2026-11-10T22:35:57+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/shift_out.json (8217d9aa25347b4b67b3b7b5b4e55f28a278bce758511fecdc7075554c0f32b8)
- Seeds: 17
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/shift_out.json --out C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-stage-001/shift_out`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Recheck candidate retention with a force-limited vertical stage and the same slip-to-preload regulator as the live G1.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold passed. Peak lift was 0.1409 m. 9 of 12 predicted holds cleared the height gate. The live run used rank 2.

## Regressions observed

The selector recovered a previous failure.

## Interruptions and failures

The run completed without interruption.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-stage-001/shift_out (dir, sha256 4c384df4911882b285fda4d4f3458bddd2d06493f839ff06aea819cc8dbb682e)
