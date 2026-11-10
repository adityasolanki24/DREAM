# Run report: Use the articulated pinch envelope and retain vertical clearance during correction

- Run ID: 2026-11-10-008/20261110-215827-clearance-alignment
- Parent task: 2026-11-10-008
- Started: 2026-11-10T21:58:27+11:00
- Ended: 2026-11-10T21:59:39+11:00
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
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-box-feedback-002`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Use the pinch envelope and preserve clearance during observed alignment correction.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Horizontal correction produced an opposing pinch on the box. The probe lost thumb contact, and the observed box rose only 0.4 mm. No full lift was requested.

## Regressions observed

The table-directed lift case remains blocked. Contact on the box advances to the probe.

## Interruptions and failures

The box stayed supported during the probe.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-box-feedback-002 (dir, sha256 02cc004a1defdb1d813f276b94bc8ed64e08e01d4ac99de3dedfcc8c7acbe571)
