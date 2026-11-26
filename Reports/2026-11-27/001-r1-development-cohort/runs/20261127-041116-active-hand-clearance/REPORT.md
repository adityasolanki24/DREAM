# Run report: Test working-hand-only support exceptions on original failures and independent cases

- Run ID: 2026-11-27-001/20261127-041116-active-hand-clearance
- Parent task: 2026-11-27-001
- Started: 2026-11-27T04:11:16+11:00
- Ended: 2026-11-27T04:22:35+11:00
- Git commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_r1_development.json (ad756e6bae64358fafcdb318e8b38036294ccf6c0765dd6b302b8a5199ac18a1); experiments/configs/g1_walk_grasp.json (a89ceed5f1503ba183356877548120b509a109f6f2da6f68eb6e8e92211f82a6); results/g1_wbc_envelope/envelope.json (d2b642f02ab8c6c2d84f7f03e425eaf6727e9f9fe2df55a24d52f7fe3228cdcf)
- Seeds: 1001,1004,1002 original failures; 83 control; 1100 fresh development box
- Model and checkpoint: Cached NVIDIA WBC walk and balance ONNX
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 with Dex3 hands
- Robot controller: Working-hand-only support exception, bounded hold feedback, independent material priors
- Hardware: Local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-active-hand-clearance-20261127/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 5
- Cases completed: 5

## Purpose

Verify the original cylinder and mug failures, an independent sphere failure, a walking control, and a fresh box after restricting support exceptions to the working hand.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
All five episodes complete. The walking control retains its verified hold. The sphere now reaches closure but fails opposing contact. Cylinder and mug still fail before closure, with changed selected stances. Fresh development box 1100 reaches closure and loses grip during lift. All five walks reach the stance with zero table-contact samples. Sensed and independent outcomes agree on all five.

## Regressions observed

The checked walking control is retained. The geometric defect is fixed, but the full physical approach problem is unresolved. The separate full fast suite passes 1017 tests, with two skipped and 91 slow tests deselected.

## Interruptions and failures

No interruption or exception. Four negative physical outcomes are preserved. This targeted diagnostic is not an estimate of the full R1 success rate.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-active-hand-clearance-20261127 (dir, sha256 e669f2462af4bff84ab07707e0891612072d8dd06f6095b521e6caa6827a6033)
