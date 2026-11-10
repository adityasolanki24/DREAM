# Run report: R1 fixed loaded wrist orientation on rectangular

- Run ID: 2026-11-10-010/20261110-223918-wrist-rectangular
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:39:18+11:00
- Ended: 2026-11-10T22:39:32+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/rectangular.json (1c4d6233c7449ab3e49c5c493ef02360f4f79153612aeaa4947af623ebfdeb45)
- Seeds: 37
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/rectangular.json --out C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/rectangular`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check whether preserving the initial loaded wrist orientation improves the static-ranking execution.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold failed. Peak lift was 0.0020 m.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The live controller stopped for grip lost.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/rectangular (dir, sha256 9ebe4e46cfb2e28aae9ddf1b438a53e09d7ee2d3ec2692651e2d6a65ce2379a2)
