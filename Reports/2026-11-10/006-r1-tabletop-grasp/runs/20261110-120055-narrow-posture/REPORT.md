# Run report: Test an interior near-nominal height band

- Run ID: 2026-11-10-006/20261110-120055-narrow-posture
- Parent task: 2026-11-10-006
- Started: 2026-11-10T12:00:55+11:00
- Ended: 2026-11-10T12:01:33+11:00
- Git commit: 6299f156b07785d65615fba43a642e1c6a2be237 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC Balance and Walk
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC near nominal height with zero torso command
- Hardware: Local Windows CPU
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-nominal-posture-002/probe.py --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-nominal-posture-002/episode`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test the same raise with a narrow interior height band and zero torso orientation command.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The plan executed, but the clearance guard stopped the raise at 2.72 s with 0.81 mm estimated separation.

## Regressions observed

The arm path still lost table clearance with negligible height change and fixed torso command.

## Interruptions and failures

Clearance stop before closing fingers.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-nominal-posture-002 (dir, sha256 ea1d14061d31c528072c545b3caf1ff1f0431aeaf956e57fd75f9a4a6a0e7b5f)
