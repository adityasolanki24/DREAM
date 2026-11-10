# Run report: Repeat the vertical raise and capture the preclose measurements

- Run ID: 2026-11-10-008/20261110-215628-measured-raise-summary
- Parent task: 2026-11-10-008
- Started: 2026-11-10T21:56:28+11:00
- Ended: 2026-11-10T21:57:20+11:00
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
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-measured-raise-002`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Confirm the measured raise and capture alignment and clearance at the end of approach.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Raise, pregrasp and reach completed. The hand had 18.8 mm of table clearance but overshot the requested grip by 53.5 mm along x. The closure guard refused.

## Regressions observed

No table clearance stop. There is a new observed hand alignment error to correct.

## Interruptions and failures

No closure or lift was commanded.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-measured-raise-002 (dir, sha256 871ace284eb3797ac5823e16801e8d327f8a5cbac57cd6dd5be17caf7e060033)
