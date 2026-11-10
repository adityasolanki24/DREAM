# Run report: Freeze and test independent development fixture lateral

- Run ID: 2026-11-10-008/20261110-220406-independent-lateral
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:04:06+11:00
- Ended: 2026-11-10T22:04:21+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_lateral.json (375844721b083e09397a58b271f342269ee0b1099d561aabc689fa7086555caf)
- Seeds: 23
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen geometry and tactile grasp logic
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_lateral.json --out C:/Users/24adi/DREAM-r1-evidence/r1-independent-lateral-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test one frozen development variation in box position, size or mass using the unchanged controller.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Passed the independent three-second grasp gate, with 146.8 mm peak lift and no protective stop.

## Regressions observed

The independently measured result is reported as development evidence.

## Interruptions and failures

See the stage and stop in Results and summary.json.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-independent-lateral-001 (dir, sha256 f592cd43d3d10abba467ad280a0c542467b5116eac07ba5ed5ed0f5a4e7cb2f8)
