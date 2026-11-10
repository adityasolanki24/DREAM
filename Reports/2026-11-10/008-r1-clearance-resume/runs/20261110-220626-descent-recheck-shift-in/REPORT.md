# Run report: Recheck shift_in after adaptive descent

- Run ID: 2026-11-10-008/20261110-220626-descent-recheck-shift-in
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:06:26+11:00
- Ended: 2026-11-10T22:06:37+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_shift_in.json (78131db7d5cdcee33ae187e74cc43faed9ce8b2a9ee621b0fe98a1a50ff580f9)
- Seeds: 11
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen adaptive descent and grip controller
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_shift_in.json --out C:/Users/24adi/DREAM-r1-evidence/r1-final-shift_in-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check the unchanged adaptive descent on this development fixture.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The approach reached closure, but the early near-contact criterion accepted a hand above the grasp depth. The thumb did not sustain load. No lift.

## Regressions observed

This early near-contact criterion regressed the nominal and lateral controls.

## Interruptions and failures

Closure ended without sustained opposing contact.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-final-shift_in-001 (dir, sha256 7cdb03825d3988394d166f6bbf772fed9facbfae4ae53ab998b04119977d605a)
