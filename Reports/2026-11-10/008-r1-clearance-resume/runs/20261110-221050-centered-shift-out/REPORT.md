# Run report: Test center-depth tracking on shift_out

- Run ID: 2026-11-10-008/20261110-221050-centered-shift-out
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:10:50+11:00
- Ended: 2026-11-10T22:11:14+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_shift_out.json (6726d0e381ba652a17b281f6538e558b18dd1a4b3adb57dbb9f8f4ce54260535)
- Seeds: 17
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen center-depth controller
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_shift_out.json --out C:/Users/24adi/DREAM-r1-evidence/r1-centered-shift_out-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Validate the measured final grasp depth and the hold gate on this development fixture.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Lifted 140.9 mm, then slipped during hold. The independent gate failed.

## Regressions observed

Nominal and lateral passes were restored relative to the early-acceptance version. See Results for this case.

## Interruptions and failures

Execution completed. See Results for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-centered-shift_out-001 (dir, sha256 664cfcf5d548cdde8e0180f3ec263b427ad660232e3bd255860f967bfed7464c)
