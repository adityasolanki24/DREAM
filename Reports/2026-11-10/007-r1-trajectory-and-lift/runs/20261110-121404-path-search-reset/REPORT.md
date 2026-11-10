# Run report: Search after initialising preview odometry

- Run ID: 2026-11-10-007/20261110-121404-path-search-reset
- Parent task: 2026-11-10-007
- Started: 2026-11-10T12:14:04+11:00
- Ended: 2026-11-10T12:14:57+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC with trajectory selection
- Hardware: Local Windows CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-trajectory-search-002`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test whether changing joint order alone gives clearance to the existing endpoint.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
All 14 transitions were rejected in preview. Eleven reached the table clearance boundary, two triggered wrist torque limits, and one triggered the touch force limit. The live hand remained at the settled pose.

## Regressions observed

No live trajectory was executed after the failed predictions.

## Interruptions and failures

Changing only the timing and joint order did not produce a feasible transition.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-trajectory-search-002 (dir, sha256 9b31646e910703a9a6b612bacc48e4ed47cdf27f21ae18c0cba40e34964b3b93)
