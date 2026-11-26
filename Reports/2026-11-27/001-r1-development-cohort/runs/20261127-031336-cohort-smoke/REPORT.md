# Run report: Smoke test the complete R1 cohort pipeline on each object family

- Run ID: 2026-11-27-001/20261127-031336-cohort-smoke
- Parent task: 2026-11-27-001
- Started: 2026-11-27T03:13:36+11:00
- Ended: 2026-11-27T03:22:27+11:00
- Git commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_r1_development.json (ad756e6bae64358fafcdb318e8b38036294ccf6c0765dd6b302b8a5199ac18a1); experiments/configs/g1_walk_grasp.json (a89ceed5f1503ba183356877548120b509a109f6f2da6f68eb6e8e92211f82a6)
- Seeds: 1000-1004 development
- Model and checkpoint: Cached NVIDIA WBC walk and balance ONNX
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 with Dex3 hands
- Robot controller: Decoupled WBC with supervised posture and grasp
- Hardware: Local CPU, two workers
- Command: `python -m experiments.g1_r1_development --config experiments/configs/g1_r1_development.json --out C:/Users/24adi/DREAM-r1-evidence/r1-cohort-smoke-20261127 --limit 5 --workers 2`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 5
- Cases completed: 5

## Purpose

Exercise the full runner on one interleaved development instance per requested object family before a 50-episode campaign.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
All five episodes completed. One bottle passed. The box gave a false sensed success at 0.0942534537 m complete clearance. Cylinder, sphere, and mug failed before closure. The 50-episode development gate is unmeasured.

## Regressions observed

The smoke introduces new development conditions. It reveals a false positive in the previous centre-displacement success predicate.

## Interruptions and failures

No run interruption or exception. Failures occur at lift/hold (box), reach (cylinder), raise (sphere), and pregrasp (mug). These phase labels do not by themselves establish root cause.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-cohort-smoke-20261127 (dir, sha256 c16220a9819d7c38216e240ca8e3435d9b37b5945bed673261e93599f843e03c)
