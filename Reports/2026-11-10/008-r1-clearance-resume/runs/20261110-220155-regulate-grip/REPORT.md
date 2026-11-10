# Run report: Maintain opposing contact while lifting and holding

- Run ID: 2026-11-10-008/20261110-220155-regulate-grip
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:01:55+11:00
- Ended: 2026-11-10T22:02:57+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC with tactile grip regulation
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-grip-regulation-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Maintain opposing touch throughout the lift and hold.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The box reached 113.8 mm and was held for about two seconds before grip loss. The full three-second gate failed.

## Regressions observed

No table or joint stop. Tactile force feedback prevented the earlier slip during lift but did not prevent later loss.

## Interruptions and failures

The grip-loss guard stopped after the object fell.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-grip-regulation-001 (dir, sha256 e29f329dd730aabe283cf1f2558500c72e226974d300129b6f8ee5d756ae5e76)
