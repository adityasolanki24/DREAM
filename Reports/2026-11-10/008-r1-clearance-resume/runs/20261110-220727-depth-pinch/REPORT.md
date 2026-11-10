# Run report: Check explicit grasp depth on pinch

- Run ID: 2026-11-10-008/20261110-220727-depth-pinch
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:07:27+11:00
- Ended: 2026-11-10T22:07:52+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen pose and tactile grasp controller
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-depth-pinch-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check the descent with an explicit depth acceptance window.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Passed the three-second hold at a 149.8 mm peak.

## Regressions observed

The hold result is reported for this development case.

## Interruptions and failures

See the grip and hold outcome in Results.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-depth-pinch-001 (dir, sha256 7ecd1891d95c1ba18b23612f14dc37bcc1b714387e86182c664129e74e1b1869)
