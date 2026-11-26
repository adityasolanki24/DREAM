# Run report: Measure the full 50-episode R1 development gate

- Run ID: 2026-11-27-001/20261127-033602-development-50
- Parent task: 2026-11-27-001
- Started: 2026-11-27T03:36:02+11:00
- Ended: 2026-11-27T03:47:52+11:00
- Git commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_r1_development.json (ad756e6bae64358fafcdb318e8b38036294ccf6c0765dd6b302b8a5199ac18a1); experiments/configs/g1_walk_grasp.json (a89ceed5f1503ba183356877548120b509a109f6f2da6f68eb6e8e92211f82a6)
- Seeds: 1000-1049 development, 10 per shape
- Model and checkpoint: Cached NVIDIA WBC walk and balance ONNX
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 with Dex3 hands
- Robot controller: Decoupled WBC with observed enclosing geometry
- Hardware: Local CPU, two campaign workers plus one regression worker
- Command: `python -m experiments.g1_r1_development --config experiments/configs/g1_r1_development.json --out C:/Users/24adi/DREAM-r1-evidence/r1-development-50-20261127 --workers 2`
- Exit status: 1
- Completed (yes/no/partial): no (interrupted)
- Cases attempted: Not recorded
- Cases completed: 6

## Purpose

Measure the preregistered 50-episode R1 development gate across all five object families.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Six completed case outputs survive. Box 1000 and bottle 1003 pass. Cylinder 1001, sphere 1002, mug 1004, and box 1005 fail during approach. The complete 50-case gate remains unmeasured.

## Regressions observed

Concurrent full testing found failures at the existing shift-in and unrotated walking regression nodes. Their causes require focused reproduction before a fresh full campaign.

## Interruptions and failures

The evaluation was deliberately stopped to investigate those regressions. Parent and child PIDs were identified by this run's exact command path and terminated. Its handle then returned exit 1. All partial files are preserved. Fifty cases were scheduled, and the exact number started is unrecorded.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-development-50-20261127 (dir, sha256 864ab98b5585a8ae0fe1b1bfbf03da0d9bd0ff0d3402ba798e5b8ba2bfaab3ea)
