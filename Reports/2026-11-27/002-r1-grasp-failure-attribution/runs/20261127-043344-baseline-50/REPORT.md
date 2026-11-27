# Run report: Measure the complete adopted development distribution on the untouched controller

- Run ID: 2026-11-27-002/20261127-043344-baseline-50
- Parent task: 2026-11-27-002
- Started: 2026-11-27T04:33:44+11:00
- Ended: 2026-11-27T13:59:03+11:00
- Git commit: 96964b8b013cf4f28b570520d491d9c9702dd20d (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_r1_development.json (ad756e6bae64358fafcdb318e8b38036294ccf6c0765dd6b302b8a5199ac18a1); experiments/configs/g1_walk_grasp.json (a89ceed5f1503ba183356877548120b509a109f6f2da6f68eb6e8e92211f82a6)
- Seeds: development 1000-1049
- Model and checkpoint: cached NVIDIA Decoupled WBC ONNX
- Simulator and version: MuJoCo 3.16.0
- Renderer: none
- Robot embodiment: free-standing G1 Dex3
- Robot controller: 96964b8
- Hardware: local CPU, two workers
- Command: `python -m experiments.g1_r1_development --config experiments/configs/g1_r1_development.json --workers 2 --out C:/Users/24adi/DREAM-r1-evidence/r1-96964b8-development-50`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 50
- Cases completed: 50

## Purpose

Measure all five families on the pre-change controller with independent scoring and count the stages where execution stops.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
All 50 cases completed. Two verified holds (bottle 1003 and mug 1024), zero false positives or false negatives. The 35-hold target failed. Of 29 pregrasp losses, 22 included hand-clearance rejection, 19 torque rejection and 16 contact-force rejection in forecasts. Categories overlap.

## Regressions observed

This is the first complete cohort. Prior partial and matched experiments do not establish a comparable population rate.

## Interruptions and failures

Exit 0. Forty-eight physical failures remain in the denominator. The execution used a clean detached checkout of 96964b8 recorded by the runner. The report tool ran in the development worktree.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-96964b8-development-50 (dir, sha256 51a222003479e313b97137e8fad23807e817e8e16233f6f76982bd57ac015c6e)
- C:/Users/24adi/DREAM-r1-evidence/r1-96964b8-development-50.log (file, sha256 a67c9a0b974083d56a722b379b5d4514b3bb6bd49675e47dc1d7787a7b7ea339)
