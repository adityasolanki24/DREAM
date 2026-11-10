# Run report: R1 physical rehearsal and live replay of fresh_near

- Run ID: 2026-11-10-010/20261110-223313-rehearsal-fresh-near
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:33:13+11:00
- Ended: 2026-11-10T22:33:28+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/fresh_near.json (45ecd64aa9814fdf0a8c6ce034a28cdcff04f725fb7c5f706bb0b317fcaabff6)
- Seeds: 29
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/fresh_near.json --out C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/fresh_near`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Screen the top geometry candidates with a prescribed wrist, then replay the selection on the live free-standing G1.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold passed. Peak lift was 0.1357 m. 0 of 12 predicted holds cleared the height gate. The live run used rank 0.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The run completed without interruption.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-rehearsal-screen-001/fresh_near (dir, sha256 0bc9cc357c4c41f383e2abb40ec3df7e6d83959811ca74fbc1c75594d8527dc7)
