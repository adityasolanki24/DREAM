# Run report: Validate bounded sensor confirmation and material isolation on five cases

- Run ID: 2026-11-27-001/20261127-040233-clearance-confirmation
- Parent task: 2026-11-27-001
- Started: 2026-11-27T04:02:33+11:00
- Ended: 2026-11-27T04:09:48+11:00
- Git commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_shift_in.json (78131db7d5cdcee33ae187e74cc43faed9ce8b2a9ee621b0fe98a1a50ff580f9); experiments/configs/g1_walk_grasp.json (a89ceed5f1503ba183356877548120b509a109f6f2da6f68eb6e8e92211f82a6); experiments/configs/g1_r1_development.json (ad756e6bae64358fafcdb318e8b38036294ccf6c0765dd6b302b8a5199ac18a1)
- Seeds: Original shift-in seed, 83, 1000, 1003, 89; all development
- Model and checkpoint: Cached NVIDIA WBC walk and balance ONNX
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 with Dex3 hands
- Robot controller: Bounded hold verification with independent forecast materials in both modes
- Hardware: Local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-clearance-confirmation-20261127/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 5
- Cases completed: 5

## Purpose

Validate bounded contact confirmation and independent material models while retaining previous holds.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Four of five selected cases pass. Shift-in, walking cube, rotated walking cube, and bottle are retained. The large box reports failure after a height correction and subsequent contact/clearance loss. There are no false success claims.

## Regressions observed

The two previously lost physical holds remain recovered. The large-box challenge remains a failed development case and is counted as such.

## Interruptions and failures

No process interruption. Negative results are preserved in the per-case records.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-clearance-confirmation-20261127 (dir, sha256 ca0d88d5e53ccc094e4e403ff48e047607a60f7d248d3a81bd0d43cb684f55ae)

## Amendments

### 2026-11-27T04:20:19+11:00

Runtime source reconstruction is stored separately at C:/Users/24adi/DREAM-r1-evidence/r1-clearance-confirmation-20261127-source. Manifest SHA-256 is 7e53e94f92fcd5a2637ee1880cecd928817ab0703c0c2bc4ae9483f3e871d94b. It restores the pre-fix planner from the completed geometry audit. This is labelled reconstructed evidence. Original run outputs remain intact.
