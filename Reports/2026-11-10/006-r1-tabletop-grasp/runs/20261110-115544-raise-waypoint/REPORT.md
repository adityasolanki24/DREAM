# Run report: Raise the hand outside the table footprint before the pinch approach

- Run ID: 2026-11-10-006/20261110-115544-raise-waypoint
- Parent task: 2026-11-10-006
- Started: 2026-11-10T11:55:44+11:00
- Ended: 2026-11-10T11:56:37+11:00
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
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-raise-waypoint-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Separate the initial raising motion from the lateral approach by deriving a waypoint outside the table footprint.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The raise waypoint was accepted, but the actual body-hand trajectory failed the clearance guard at 2.72 s. No closure or lift.

## Regressions observed

No constraint or force limits changed. The waypoint alone did not fix the trajectory.

## Interruptions and failures

Stopped on hand clearance during raise.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-raise-waypoint-001 (dir, sha256 f1df1fb30f42c28f3f4a75d66f620fba9d7f360a93c42dcfc810e1525a6a774b)
