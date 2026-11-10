# Run report: Attempt a monitored pinch with articulated hand clearance

- Run ID: 2026-11-10-006/20261110-115504-articulated-guard
- Parent task: 2026-11-10-006
- Started: 2026-11-10T11:55:04+11:00
- Ended: 2026-11-10T11:55:41+11:00
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
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-articulated-guard-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Replace the hand proxy with the articulated fixture check and test the pregrasp trajectory.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The pregrasp goal was feasible, but the trajectory brought the measured hand within 1.08 mm of the tabletop at 2.06 s. The clearance guard stopped it before closure.

## Regressions observed

No change in force or torque limits. The previously possible table-directed lift did not occur.

## Interruptions and failures

The direct transition to the feasible endpoint lacked clearance during motion.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-articulated-guard-001 (dir, sha256 e485d1aa26c2599c9e95cb8dca26c510444012cfb7b68a723eacaaf233172d9b)
