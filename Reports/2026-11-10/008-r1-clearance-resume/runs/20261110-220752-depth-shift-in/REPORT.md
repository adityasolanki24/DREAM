# Run report: Check explicit grasp depth on shift_in

- Run ID: 2026-11-10-008/20261110-220752-depth-shift-in
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:07:52+11:00
- Ended: 2026-11-10T22:08:12+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_shift_in.json (78131db7d5cdcee33ae187e74cc43faed9ce8b2a9ee621b0fe98a1a50ff580f9)
- Seeds: 11
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen pose and tactile grasp controller
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_shift_in.json --out C:/Users/24adi/DREAM-r1-evidence/r1-depth-shift_in-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check the descent with an explicit depth acceptance window.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Lifted to 142.6 mm, then lost grip before the hold gate.

## Regressions observed

The hold result is reported for this development case.

## Interruptions and failures

See the grip and hold outcome in Results.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-depth-shift_in-001 (dir, sha256 f02f526f35d41ad2d3dafa2bb18a5c8effb4493eed66971f1159ef4137c15725)
