# Run report: Reproduce the two failures found by the full regression suite

- Run ID: 2026-11-27-001/20261127-034718-clearance-regressions
- Parent task: 2026-11-27-001
- Started: 2026-11-27T03:47:18+11:00
- Ended: 2026-11-27T03:51:47+11:00
- Git commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_shift_in.json (78131db7d5cdcee33ae187e74cc43faed9ce8b2a9ee621b0fe98a1a50ff580f9); experiments/configs/g1_walk_grasp.json (a89ceed5f1503ba183356877548120b509a109f6f2da6f68eb6e8e92211f82a6)
- Seeds: Fixed seeds in the two saved development configurations
- Model and checkpoint: Cached NVIDIA WBC walk and balance ONNX
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 with Dex3 hands
- Robot controller: Current full-surface clearance feedback
- Hardware: Local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-clearance-regressions-20261127/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 2
- Cases completed: 2

## Purpose

Reproduce the two failed existing regressions on the exact current source and preserve full traces.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Both are physical failures. Shift-in selects rank 61 and loses object contact during hold after 130 samples. The walking case rejects the previously successful predicted grasp after grip loss, falls back to rank 0, and stops on object displacement before closure.

## Regressions observed

Both existing successes are lost. The new fixed 2 cm margin above complete surface clearance drives additional lifting and changes the forecast outcome and fallback choice.

## Interruptions and failures

Both diagnostic episodes completed. The full suite failures were not merely overly specific assertions. Their source and configurations remain preserved.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-clearance-regressions-20261127 (dir, sha256 bd9ec9db03a589aad16412d5c5ce7c0278a371181876556970e7b0cf42a81c49)
