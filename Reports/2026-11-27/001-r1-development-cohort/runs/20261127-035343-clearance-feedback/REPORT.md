# Run report: Validate bounded clearance feedback and uninterrupted holds

- Run ID: 2026-11-27-001/20261127-035343-clearance-feedback
- Parent task: 2026-11-27-001
- Started: 2026-11-27T03:53:43+11:00
- Ended: 2026-11-27T04:00:32+11:00
- Git commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_shift_in.json (78131db7d5cdcee33ae187e74cc43faed9ce8b2a9ee621b0fe98a1a50ff580f9); experiments/configs/g1_walk_grasp.json (a89ceed5f1503ba183356877548120b509a109f6f2da6f68eb6e8e92211f82a6); experiments/configs/g1_r1_development.json (ad756e6bae64358fafcdb318e8b38036294ccf6c0765dd6b302b8a5199ac18a1)
- Seeds: Original shift-in seed, 83, 1000, 1003, 89; all development
- Model and checkpoint: Cached NVIDIA WBC walk and balance ONNX
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 with Dex3 hands
- Robot controller: Bounded clearance feedback with a fresh continuous 3 s hold after correction
- Hardware: Local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-clearance-feedback-20261127/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 5
- Cases completed: 5

## Purpose

Recover the two lost holds while retaining complete observed clearance and validating independent 3 s hold windows.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Four of five physical cases pass. Shift-in and both walking cube frames are restored, and the bottle is retained. The large box corrects its height once but stops verification at the first zero-thumb-force sample after 106 subsequent hold samples. It reports failure accurately.

## Regressions observed

The two original regressions are recovered. Large box 1000 remains unresolved in this candidate controller. Its final sample has thumb force 0 N, opposing force 0.8896 N, encoder fraction 0.6422, and clearance 0.1108801865 m.

## Interruptions and failures

No process interruption. The large box ends at the first unconfirmed tactile sample before the existing 0.10 s contact-loss guard can establish recovery or loss. A bounded verification restart will be tested next.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-clearance-feedback-20261127 (dir, sha256 ee139ff53308f15be481f36a22debbe0cab217aae30e2ced5250071c7f2c989e)
