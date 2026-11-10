# Run report: Freeze and test independent development fixture shift_out

- Run ID: 2026-11-10-008/20261110-220355-independent-shift-out
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:03:55+11:00
- Ended: 2026-11-10T22:04:06+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_shift_out.json (6726d0e381ba652a17b281f6538e558b18dd1a4b3adb57dbb9f8f4ce54260535)
- Seeds: 17
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen geometry and tactile grasp logic
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_shift_out.json --out C:/Users/24adi/DREAM-r1-evidence/r1-independent-shift_out-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test one frozen development variation in box position, size or mass using the unchanged controller.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Descent completed, but the hand was 41.5 mm above the grip target. The pinch-envelope check refused closure.

## Regressions observed

The independently measured result is reported as development evidence.

## Interruptions and failures

See the stage and stop in Results and summary.json.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-independent-shift_out-001 (dir, sha256 1d04b13dd33e0bb5d675aed2c59723ebf7b8d2896abfd9f003640ad7b78886f3)
