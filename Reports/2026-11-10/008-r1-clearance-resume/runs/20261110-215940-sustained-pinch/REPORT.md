# Run report: Test sustained tactile pinch and relative vertical probe

- Run ID: 2026-11-10-008/20261110-215940-sustained-pinch
- Parent task: 2026-11-10-008
- Started: 2026-11-10T21:59:40+11:00
- Ended: 2026-11-10T22:00:43+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC with tactile pinch and relative vertical lift
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-sustained-pinch-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test sustained tactile loading and a vertical tool-relative probe.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Sustained contact was reached at closure fraction 0.86. Peak box movement was 8.1 mm during closure. The vertical probe lost thumb contact and produced only 0.33 mm net rise.

## Regressions observed

The probe guard still refused a full lift.

## Interruptions and failures

The grasp slipped during probe.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-sustained-pinch-001 (dir, sha256 5de8de6f9f62049cccc8957c3ddf2da9377d36eeb0087c2b3387e981e6aa8b14)
