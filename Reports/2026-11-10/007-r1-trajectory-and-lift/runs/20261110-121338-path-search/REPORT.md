# Run report: Search for a clear joint transition using sensor-initialised rollouts

- Run ID: 2026-11-10-007/20261110-121338-path-search
- Parent task: 2026-11-10-007
- Started: 2026-11-10T12:13:38+11:00
- Ended: 2026-11-10T12:14:02+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC with trajectory selection
- Hardware: Local Windows CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-trajectory-search-001`
- Exit status: 1
- Completed (yes/no/partial): no (crashed)
- Cases attempted: 1
- Cases completed: 0

## Purpose

Test sensor-initialised rehearsal before the raise.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
No physical result. Preview crashed before any new live reach was attempted.

## Regressions observed

Not measured.

## Interruptions and failures

The planning environment had not called reset to initialise the odometry bias.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-trajectory-search-001 (dir, sha256 53f51953f66652aaf45c4e7793b5f98af558655dce970540c559ab88a912efae)
