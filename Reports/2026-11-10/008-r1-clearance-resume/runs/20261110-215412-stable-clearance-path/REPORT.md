# Run report: Search from measured balance stability with clearance-ranked waypoints

- Run ID: 2026-11-10-008/20261110-215412-stable-clearance-path
- Parent task: 2026-11-10-008
- Started: 2026-11-10T21:54:12+11:00
- Ended: 2026-11-10T21:55:44+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC with clearance-ranked path search
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-stable-path-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Search geometric waypoints after measured settling and prioritize elevation before reaching.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Settling passed but no raise candidate passed preview. No new arm motion was executed.

## Regressions observed

Not measured. Live approach did not execute.

## Interruptions and failures

The candidate set still changed hand orientation while raising.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-stable-path-001 (dir, sha256 1b3eab21d8fe6dcf5cc8f9c9cafeaf09c5edf40144f75571cf8208a6e69d5ba3)
