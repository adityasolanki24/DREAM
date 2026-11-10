# Run report: Align above the object before the vertical descent

- Run ID: 2026-11-10-008/20261110-220044-align-above-box
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:00:44+11:00
- Ended: 2026-11-10T22:01:54+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC with staged alignment and tactile closure
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-overhead-alignment-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Align horizontally above the box and descend from the measured pose.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The box rose 24.3 mm in the probe and reached 70.0 mm during lift. It then slipped as the measured thumb and finger forces decreased. The independent hold gate did not pass.

## Regressions observed

No tabletop stop. Grasp loss became the limiting physical failure.

## Interruptions and failures

The object fell to the table before hold.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-overhead-alignment-001 (dir, sha256 f62944656340ed9ef446950596c969ccea01c5b1d0e9dcd015f34d79ac90eb98)
