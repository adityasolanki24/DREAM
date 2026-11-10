# Run report: Test an unseen centre and mass for the fixed-width box

- Run ID: 2026-11-10-008/20261110-221219-fresh-far
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:12:19+11:00
- Ended: 2026-11-10T22:12:47+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_fresh_far.json (ef003711da9ddb4d4c7692ef750c102396e37dcaa59e4bcca61f7c8c8853f8be)
- Seeds: 31
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen center-depth controller
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_fresh_far.json --out C:/Users/24adi/DREAM-r1-evidence/r1-untuned-fresh_far-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test an untuned box position and mass after freezing the control code.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"fresh_far","preclose":true,"stop":"grip lost","peak_m":0.14072232178522381,"verified":false}

## Regressions observed

This fixture had no earlier pass to compare.

## Interruptions and failures

Execution completed. See the trace and Results for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-untuned-fresh_far-001 (dir, sha256 dcdf9b56b15abcf9883818fac3853cae7d9c584c4bb2d26d6f4875a7fdf071a0)
