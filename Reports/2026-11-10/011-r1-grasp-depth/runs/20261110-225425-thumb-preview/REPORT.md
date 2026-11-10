# Run report: Verify combined depth and contact closure selection on the same development cases

- Run ID: 2026-11-10-011/20261110-225425-thumb-preview
- Parent task: 2026-11-10-011
- Started: 2026-11-10T22:54:25+11:00
- Ended: 2026-11-10T22:57:14+11:00
- Git commit: c04ffd81e7a2adffeb84f65a47e64afee342467e (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: results/g1_r1_contact_rank_dev_20261110/summary.json (7ae701b1b4f3fd899c559cfca9586f25dde72bb2819d42cdb6975603b4964b70)
- Seeds: same nine live development cases
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU, 2 workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-depth-thumb-preview-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 9
- Cases completed: 9

## Purpose

Verify the expanded selector including alternative thumb closure profiles on all nine matched live cases.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Eight of nine cases passed. The cylinder held with the deeper standard grip. The inward cube held with altered thumb closure. The sphere lost opposition in the initial probe.

## Regressions observed

All six prior verified cases passed. The depth-only cylinder gain was preserved.

## Interruptions and failures

All nine runs completed. Sphere retention remains an open failure.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-depth-thumb-preview-001 (dir, sha256 84f5f19828806effbd990cf7ab7d227340b980a6c0dd3c99d7f052341160ca45)
