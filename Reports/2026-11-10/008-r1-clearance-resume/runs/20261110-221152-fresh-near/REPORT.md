# Run report: Test an unseen centre and mass for the fixed-width box

- Run ID: 2026-11-10-008/20261110-221152-fresh-near
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:11:52+11:00
- Ended: 2026-11-10T22:12:16+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_fresh_near.json (b2c9615712423608644d8a3ccb34a2b7d0c2f8a3af11c2a2c460c5e9ab8713d6)
- Seeds: 29
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen center-depth controller
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_fresh_near.json --out C:/Users/24adi/DREAM-r1-evidence/r1-untuned-fresh_near-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test an untuned box position and mass after freezing the control code.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"fresh_near","preclose":true,"stop":null,"peak_m":0.14831210517173443,"verified":true}

## Regressions observed

This fixture had no earlier pass to compare.

## Interruptions and failures

Execution completed. See the trace and Results for the physical outcome.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-untuned-fresh_near-001 (dir, sha256 9321d32a389347e6bfc9a79cf7f7920dfb858273a5ed380c20a9f6a60a384a86)
