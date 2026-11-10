# Run report: Test center-depth tracking on lateral

- Run ID: 2026-11-10-008/20261110-220950-centered-lateral
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:09:50+11:00
- Ended: 2026-11-10T22:10:13+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_lateral.json (375844721b083e09397a58b271f342269ee0b1099d561aabc689fa7086555caf)
- Seeds: 23
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen center-depth controller
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_lateral.json --out C:/Users/24adi/DREAM-r1-evidence/r1-centered-lateral-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Validate the measured final grasp depth and the hold gate on this development fixture.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Passed the independent three-second hold, peak 146.8 mm.

## Regressions observed

Nominal and lateral passes were restored relative to the early-acceptance version. See Results for this case.

## Interruptions and failures

Execution completed. See Results for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-centered-lateral-001 (dir, sha256 89d2e567a22d7b713f7c48b298a354ea2f6872ca0f4f6223cc25091e2dfb25d2)
