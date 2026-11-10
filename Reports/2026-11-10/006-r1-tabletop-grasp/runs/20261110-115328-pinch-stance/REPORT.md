# Run report: Test the same pinch from a stance accounting for settling retreat

- Run ID: 2026-11-10-006/20261110-115328-pinch-stance
- Parent task: 2026-11-10-006
- Started: 2026-11-10T11:53:28+11:00
- Ended: 2026-11-10T11:54:13+11:00
- Git commit: 6299f156b07785d65615fba43a642e1c6a2be237 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC Balance and Walk
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC and posture supervisor
- Hardware: Local Windows CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-pinch-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Reduce reach distance while preserving table, object and limits, after measuring the settle displacement.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
At settled x=0.083 m the planned hand error dropped to 22.6 mm, still above the 20 mm planning tolerance. No motion was attempted.

## Regressions observed

Not measured. The change affects this diagnostic initial pose only.

## Interruptions and failures

The planner refused the target. The next check separates reach from the fixed hand collision approximation.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-pinch-001 (dir, sha256 619d520e48325c26ce94ab81c402c537b7a815fc06feda142d352e25916903f8)
