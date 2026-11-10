# Run report: Check explicit grasp depth on shift_out

- Run ID: 2026-11-10-008/20261110-220831-depth-shift-out
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:08:31+11:00
- Ended: 2026-11-10T22:08:48+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_shift_out.json (6726d0e381ba652a17b281f6538e558b18dd1a4b3adb57dbb9f8f4ce54260535)
- Seeds: 17
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen pose and tactile grasp controller
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_shift_out.json --out C:/Users/24adi/DREAM-r1-evidence/r1-depth-shift_out-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check the descent with an explicit depth acceptance window.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Aligned and closed, but lost grip during probe, with a 4.6 mm peak rise.

## Regressions observed

The hold result is reported for this development case.

## Interruptions and failures

See the grip and hold outcome in Results.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-depth-shift_out-001 (dir, sha256 6d361b167bc36b05710bda0db254b71ab87bb5b58674bbc63f5f3e6370bbc4f0)
