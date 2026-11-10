# Run report: Maintain pinch preload using observed object motion relative to the wrist

- Run ID: 2026-11-10-008/20261110-220258-slip-feedback
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:02:58+11:00
- Ended: 2026-11-10T22:03:43+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC with tactile and pose grip feedback
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-slip-feedback-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Use box motion relative to the wrist to adapt pinch preload before contact is lost.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The nominal case passed the independent grasp gate. Peak lift was 150.0 mm. The box stayed at least 100 mm above its start for 3 s with only right-hand contact and no protective stop.

## Regressions observed

No failure detected in this nominal physical episode. Other fixtures follow separately.

## Interruptions and failures

None.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-slip-feedback-001 (dir, sha256 b1ef36de8ee20c33ac740003a164bb2cd36ec2f841626297e41b5ab36e9cb71f)
