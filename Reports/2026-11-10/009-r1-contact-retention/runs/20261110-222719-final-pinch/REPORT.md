# Run report: Recheck pinch on the shape-generalized selector

- Run ID: 2026-11-10-009/20261110-222719-final-pinch
- Parent task: 2026-11-10-009
- Started: 2026-11-10T22:27:19+11:00
- Ended: 2026-11-10T22:27:30+11:00
- Git commit: d5cc887a41d441f7408d044d4b9e5f6e9a66e572 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/pinch.json (4f65f639a9b83212df34bff6fb78b28cd00b3fc04d0e27b350ec29db6e7d1e4b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC plus geometry-ranked pinch
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/pinch`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Verify the final shared geometry selector on the matched cube fixture.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
{"case":"pinch","stop":null,"peak":0.14223391185328682,"verified":true}

## Regressions observed

Matched comparison recorded in the aggregate evidence.

## Interruptions and failures

Execution completed. The physical outcome is recorded in Results.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-contact-replay-001/pinch (dir, sha256 6c79d41f3ec118cdd447979085b86e67fbd1fff1c49705abb2e9b3ae601f0236)
