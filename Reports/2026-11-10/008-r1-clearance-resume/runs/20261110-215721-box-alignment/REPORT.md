# Run report: Correct approach from measured hand and box pose

- Run ID: 2026-11-10-008/20261110-215721-box-alignment
- Parent task: 2026-11-10-008
- Started: 2026-11-10T21:57:21+11:00
- Ended: 2026-11-10T21:58:26+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC with measured grasp alignment
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-box-feedback-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Correct the tool position by measured alignment error rather than a stored offset.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The full correction had a predicted table penetration of 7.2 mm. It was refused before execution.

## Regressions observed

Approach and closure safeguards retained. No collision was attempted by the correction.

## Interruptions and failures

The vertical component of the correction competed with the clearance constraint.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-box-feedback-001 (dir, sha256 624ce5d732d0a4a6ea24da447a31cdd2d0891b069bc0261e027753c38eafb5eb)
