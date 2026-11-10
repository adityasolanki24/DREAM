# Run report: Test center-depth tracking on shift_in

- Run ID: 2026-11-10-008/20261110-221013-centered-shift-in
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:10:13+11:00
- Ended: 2026-11-10T22:10:31+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_shift_in.json (78131db7d5cdcee33ae187e74cc43faed9ce8b2a9ee621b0fe98a1a50ff580f9)
- Seeds: 11
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen center-depth controller
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_shift_in.json --out C:/Users/24adi/DREAM-r1-evidence/r1-centered-shift_in-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Validate the measured final grasp depth and the hold gate on this development fixture.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The measured hand remained 2.16 mm high and the guarded descent found no further improving move. No closure.

## Regressions observed

Nominal and lateral passes were restored relative to the early-acceptance version. See Results for this case.

## Interruptions and failures

Execution completed. See Results for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-centered-shift_in-001 (dir, sha256 699a08872f3cee861c5ae06bab58d8e41333c371a5aad12c9fa88bd43917b29a)
