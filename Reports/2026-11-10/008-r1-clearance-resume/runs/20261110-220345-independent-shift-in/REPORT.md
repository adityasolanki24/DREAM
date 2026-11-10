# Run report: Freeze and test independent development fixture shift_in

- Run ID: 2026-11-10-008/20261110-220345-independent-shift-in
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:03:45+11:00
- Ended: 2026-11-10T22:03:55+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_shift_in.json (78131db7d5cdcee33ae187e74cc43faed9ce8b2a9ee621b0fe98a1a50ff580f9)
- Seeds: 11
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen geometry and tactile grasp logic
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_shift_in.json --out C:/Users/24adi/DREAM-r1-evidence/r1-independent-shift_in-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test one frozen development variation in box position, size or mass using the unchanged controller.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Stopped on hand clearance during descent before closing. No verified grasp.

## Regressions observed

The independently measured result is reported as development evidence.

## Interruptions and failures

See the stage and stop in Results and summary.json.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-independent-shift_in-001 (dir, sha256 d9c59a8db073a3458676f0a36c60fce58ef728ecdb3064bc13ef7984622f7fdf)
