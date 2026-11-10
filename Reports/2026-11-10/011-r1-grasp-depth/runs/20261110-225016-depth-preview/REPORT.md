# Run report: Compare depth-diverse grasp proposals against the current R1 controller

- Run ID: 2026-11-10-011/20261110-225016-depth-preview
- Parent task: 2026-11-10-011
- Started: 2026-11-10T22:50:16+11:00
- Ended: 2026-11-10T22:52:47+11:00
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
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-depth-preview-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 9
- Cases completed: 9

## Purpose

Evaluate size-scaled front-to-back proposals and distinct backups on the nine matched live cases.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Seven of nine cases passed. The cylinder recovered while the inward cube and sphere remained failures.

## Regressions observed

All six prior verified holds were preserved.

## Interruptions and failures

All nine runs completed. The inward cube slipped after the probe and the sphere during it.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-depth-preview-001 (dir, sha256 3eef87b748efdf3760734f4ef831341ff06e2ad30c6c2c925d6f31ae7107db38)
