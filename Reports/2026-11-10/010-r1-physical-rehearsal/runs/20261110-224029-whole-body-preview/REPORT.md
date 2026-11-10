# Run report: Confirm the proposed grasp with an independent whole-body rollout

- Run ID: 2026-11-10-010/20261110-224029-whole-body-preview
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:40:29+11:00
- Ended: 2026-11-10T22:42:47+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: results/g1_r1_contact_rank_dev_20261110/summary.json (7ae701b1b4f3fd899c559cfca9586f25dde72bb2819d42cdb6975603b4964b70)
- Seeds: same nine matched cases
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU, 2 workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-whole-body-preview-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 9
- Cases completed: 9

## Purpose

Replay each proposed pose with the full simulated G1 controller before execution.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The live run passed 5/6 cubes and 1/3 additional geometries. The outward cube and rectangular box were the recovered holds.

## Regressions observed

All four previously verified holds passed. Generalisation to a new distribution was not measured.

## Interruptions and failures

The first implementation reused the live episode seed for odometry noise. The independent-noise rerun supersedes it as evidence for the selector.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-whole-body-preview-001 (dir, sha256 1ee524824877fbad4dbb404d450c72f1bb53a0f506724825becc9b9282870fd2)
