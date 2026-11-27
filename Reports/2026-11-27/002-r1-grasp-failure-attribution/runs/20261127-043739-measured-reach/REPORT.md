# Run report: Replay physically refused approaches from the same selected grasps and a walking control

- Run ID: 2026-11-27-002/20261127-043739-measured-reach
- Parent task: 2026-11-27-002
- Started: 2026-11-27T04:37:39+11:00
- Ended: 2026-11-27T04:40:19+11:00
- Git commit: 96964b8b013cf4f28b570520d491d9c9702dd20d (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_walk_grasp.json (a89ceed5f1503ba183356877548120b509a109f6f2da6f68eb6e8e92211f82a6); experiments/configs/g1_r1_development.json (ad756e6bae64358fafcdb318e8b38036294ccf6c0765dd6b302b8a5199ac18a1)
- Seeds: 1001,1004; walking control 83
- Model and checkpoint: cached NVIDIA WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: none
- Robot embodiment: G1 Dex3
- Robot controller: measured arm refinement after whole-body approach refusal
- Hardware: local CPU, one worker
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-measured-approach-replay-20261127/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 3
- Cases completed: 3

## Purpose

Test whether incremental, supervised arm motion from measured posture can recover an approach the nominal whole-body planner rejects.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Cylinder 1001 completed seven corrective arm movements, reached closure, then lost object contact. Mug 1004 completed two corrections before further forecasts exceeded the 35 N force guard. Control 83 retained the verified hold without correction.

## Regressions observed

The previously passing walking control remained successful. Fixed grasp candidates were reused for this paired diagnostic. It establishes neither a broad rate nor robustness of automatic selection.

## Interruptions and failures

All three cases ran to completion; the two failed holds remain failures.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-measured-approach-replay-20261127 (dir, sha256 43596216f0dd27915f749576598e76c81a9f96aa1e6007fd621b8ea0b8ff4a99)
