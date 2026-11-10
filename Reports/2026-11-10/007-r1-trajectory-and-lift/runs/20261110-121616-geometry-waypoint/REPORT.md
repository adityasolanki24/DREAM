# Run report: Sample intermediate joint poses screened by articulated clearance

- Run ID: 2026-11-10-007/20261110-121616-geometry-waypoint
- Parent task: 2026-11-10-007
- Started: 2026-11-10T12:16:16+11:00
- Ended: 2026-11-10T21:54:10+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC with geometric and dynamic screening
- Hardware: Local Windows CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-trajectory-search-004`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Screen sampled intermediate arm poses geometrically before running the balance policy.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
All 20 additional previewed detours stopped, mostly on hand clearance. None reached live execution.

## Regressions observed

No live approach was executed.

## Interruptions and failures

The process finished after interruption and its result was recovered in the next turn.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-trajectory-search-004 (dir, sha256 972e186db0fbe3b84bc286b90e412e84291aa4995f98d5d25b2613611b2ab42f)
