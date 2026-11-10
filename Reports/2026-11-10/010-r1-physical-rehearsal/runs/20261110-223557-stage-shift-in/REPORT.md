# Run report: R1 actuated rehearsal and live replay of shift_in

- Run ID: 2026-11-10-010/20261110-223557-stage-shift-in
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:35:57+11:00
- Ended: 2026-11-10T22:36:10+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/shift_in.json (d708b6eb9bc3f46e53db3a3a8d91b51d3530c5ad40cd1da1534cd97d21f9afce)
- Seeds: 11
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/shift_in.json --out C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-stage-001/shift_in`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Recheck candidate retention with a force-limited vertical stage and the same slip-to-preload regulator as the live G1.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold failed. Peak lift was 0.1025 m. 0 of 6 predicted holds cleared the height gate. The live run used rank 0.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The live controller stopped for grip lost.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-stage-001/shift_in (dir, sha256 91d52ae20efdce20c0abbb1aa88da575747beeb926036b58c37a8c70f970ff86)
