# Run report: Decouple hand orientation from vertical ascent

- Run ID: 2026-11-10-008/20261110-215546-measured-vertical-raise
- Parent task: 2026-11-10-008
- Started: 2026-11-10T21:55:46+11:00
- Ended: 2026-11-10T21:56:27+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC with clearance-ranked path search
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-measured-raise-001`
- Exit status: 1
- Completed (yes/no/partial): partial (partial)
- Cases attempted: 1
- Cases completed: 0

## Purpose

Raise in the observed wrist orientation, then turn toward the box.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Trace contains completed raise, pregrasp and reach stages, and thumb contact with the box at the end of reach. No closure rows.

## Regressions observed

Not measured. Full result serialization failed.

## Interruptions and failures

The summary JSON rejected the NumPy boolean from the preclose check. The physics trace was written before the error.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-measured-raise-001 (dir, sha256 f889db9f505ec606cca9faf10be09309eec0dd6d59706a17d7a284d836ffbd95)
