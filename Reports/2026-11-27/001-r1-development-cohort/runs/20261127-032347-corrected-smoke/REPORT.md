# Run report: Replay five object families with observed full-surface clearance

- Run ID: 2026-11-27-001/20261127-032347-corrected-smoke
- Parent task: 2026-11-27-001
- Started: 2026-11-27T03:23:47+11:00
- Ended: 2026-11-27T03:36:53+11:00
- Git commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_r1_development.json (ad756e6bae64358fafcdb318e8b38036294ccf6c0765dd6b302b8a5199ac18a1); experiments/configs/g1_walk_grasp.json (a89ceed5f1503ba183356877548120b509a109f6f2da6f68eb6e8e92211f82a6)
- Seeds: 1000-1004 development replay
- Model and checkpoint: Cached NVIDIA WBC walk and balance ONNX
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 with Dex3 hands
- Robot controller: Decoupled WBC with observed enclosing geometry
- Hardware: Local CPU, two workers
- Command: `python -m experiments.g1_r1_development --config experiments/configs/g1_r1_development.json --out C:/Users/24adi/DREAM-r1-evidence/r1-cohort-clearance-20261127 --limit 5 --workers 2`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 5
- Cases completed: 5

## Purpose

Replay all five smoke scenes after replacing centre rise with conservative observed complete-object clearance, in both lift control and sensed success.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
All five episodes completed. Two verified holds (box and bottle), zero false positives and zero false negatives. The box improved from a false sensed success at 0.0942534537 m minimum complete clearance to a verified hold at 0.1185410178 m over 150 hold samples. Bottle outcome is retained. This is a five-case development smoke, and the 50-episode gate remains unmeasured.

## Regressions observed

No previous smoke success was lost. The three earlier approach failures remain failures at the same stages.

## Interruptions and failures

Cylinder fails at reach, sphere at raise, and mug at pregrasp. No exceptions or interruptions occurred.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-cohort-clearance-20261127 (dir, sha256 fc9ef4c3f588f9e5a55832972b90a73e74d716b4677a5c80b919bbe158cba705)
