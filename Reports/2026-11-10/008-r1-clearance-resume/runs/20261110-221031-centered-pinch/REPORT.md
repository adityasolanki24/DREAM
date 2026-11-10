# Run report: Test center-depth tracking on pinch

- Run ID: 2026-11-10-008/20261110-221031-centered-pinch
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:10:31+11:00
- Ended: 2026-11-10T22:10:50+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen center-depth controller
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-centered-pinch-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Validate the measured final grasp depth and the hold gate on this development fixture.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Passed the independent three-second hold, peak 150.6 mm.

## Regressions observed

Nominal and lateral passes were restored relative to the early-acceptance version. See Results for this case.

## Interruptions and failures

Execution completed. See Results for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-centered-pinch-001 (dir, sha256 9ced59aa7d4ce40ab197791ce05e92a90402afa9af12e0f8d8ba27fed0674c0d)
