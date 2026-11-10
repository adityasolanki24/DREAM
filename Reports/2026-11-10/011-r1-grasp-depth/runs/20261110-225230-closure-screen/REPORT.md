# Run report: Screen depth and thumb articulation under load

- Run ID: 2026-11-10-011/20261110-225230-closure-screen
- Parent task: 2026-11-10-011
- Started: 2026-11-10T22:52:30+11:00
- Ended: 2026-11-10T22:52:51+11:00
- Git commit: c04ffd81e7a2adffeb84f65a47e64afee342467e (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_sphere.json (7765331ea21723f3cf865a23638312077868948214294e28a2c429ecd078ac14); experiments/configs/g1_grasp_bringup_shift_in.json (78131db7d5cdcee33ae187e74cc43faed9ce8b2a9ee621b0fe98a1a50ff580f9)
- Seeds: 43,11
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Dex3 rehearsal stage
- Robot controller: Force-limited vertical stage
- Hardware: local CPU
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-depth-closure-screen-001/screen.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 2
- Cases completed: 2

## Purpose

Distinguish insufficient depth proposals from a thumb contact geometry limitation on the remaining small cube and sphere.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The small cube had four stage holds at minus 15 degrees of thumb abduction and none at 0 or plus 15 degrees. The sphere produced zero stage holds at all three angles.

## Regressions observed

Live performance was not measured in this stage-only run.

## Interruptions and failures

The run completed. The selected probes did not produce a stable sphere hold.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-depth-closure-screen-001 (dir, sha256 b664ac72eb8d0ab04bb238e98faef769dead25e576159cb3415ad06b58300ebe)
