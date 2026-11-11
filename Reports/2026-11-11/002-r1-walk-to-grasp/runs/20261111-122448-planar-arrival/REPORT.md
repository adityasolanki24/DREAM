# Run report: Replay the walking starts with planar position and heading control

- Run ID: 2026-11-11-002/20261111-122448-planar-arrival
- Parent task: 2026-11-11-002
- Started: 2026-11-11T12:24:48+11:00
- Ended: 2026-11-11T12:25:46+11:00
- Git commit: c866699583059bd16b3f3364b04ffc3cd0fb2dd6 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 81,83
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC, planar odometry feedback
- Hardware: local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-walk-planar-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 2
- Cases completed: 2

## Purpose

Test forward and lateral feedback with an independently controlled final heading.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Both starts arrived. The front raise had no verified trajectory. The offset reach exceeded the planned support margin after the raise and pregrasp.

## Regressions observed

This run compared walking starts only. Fixed-stance behaviour was checked separately.

## Interruptions and failures

The arrival change resolved the first stopping condition. Neither grasp was recovered.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-walk-planar-001 (dir, sha256 63b7cdf525ed24f4ea72e83551d6787dd58c4452e9e39310f4f47b53c445b9db)
