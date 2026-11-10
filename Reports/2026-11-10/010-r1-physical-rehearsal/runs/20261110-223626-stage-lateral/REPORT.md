# Run report: R1 actuated rehearsal and live replay of lateral

- Run ID: 2026-11-10-010/20261110-223626-stage-lateral
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:36:26+11:00
- Ended: 2026-11-10T22:36:40+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/lateral.json (f71936103c18044bc5f51c1c2f78f1b0bc8111c39ae4d1a8ae1494f14cb99342)
- Seeds: 23
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/lateral.json --out C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-stage-001/lateral`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Recheck candidate retention with a force-limited vertical stage and the same slip-to-preload regulator as the live G1.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold passed. Peak lift was 0.1485 m. 2 of 12 predicted holds cleared the height gate. The live run used rank 4.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The run completed without interruption.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-stage-001/lateral (dir, sha256 d6be583ec122008a011de72a349e477efe71dbb72bf8291962531eece78743fe)
