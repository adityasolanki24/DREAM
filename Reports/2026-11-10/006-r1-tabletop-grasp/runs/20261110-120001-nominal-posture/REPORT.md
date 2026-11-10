# Run report: Separate torso posture changes from the arm trajectory

- Run ID: 2026-11-10-006/20261110-120001-nominal-posture
- Parent task: 2026-11-10-006
- Started: 2026-11-10T12:00:01+11:00
- Ended: 2026-11-10T12:00:46+11:00
- Git commit: 6299f156b07785d65615fba43a642e1c6a2be237 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC Balance and Walk
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC constrained to nominal height and zero torso command
- Hardware: Local Windows CPU
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-nominal-posture-001/probe.py --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-nominal-posture-001/episode`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Keep the WBC posture command at nominal to separate torso motion from arm motion.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The IK found a pose with 0.004 mm hand residual but the planned height exceeded the zero-width 0.74 m bound by 0.57 micrometres. Execution did not start.

## Regressions observed

Not measured. This is an external diagnostic capability restriction.

## Interruptions and failures

Zero-width height restriction exposed solver residual at the command boundary.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-nominal-posture-001 (dir, sha256 7e28b05d113f7da190d4f14641390195c8107b8303fcb5f73daf76251f38c8f0)
