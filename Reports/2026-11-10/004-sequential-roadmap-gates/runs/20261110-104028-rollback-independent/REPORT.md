# Run report: Verify correction with a shifted target and new development seed

- Run ID: 2026-11-10-004/20261110-104028-rollback-independent
- Parent task: 2026-11-10-004
- Started: 2026-11-10T10:40:28+11:00
- Ended: 2026-11-10T10:57:58+11:00
- Git commit: efd6872f4c375f948244e04c5beb71eade626925 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: Not applicable. This run uses the repository physics regression fixture or the captured probe script.
- Seeds: 0, 1, 21
- Model and checkpoint: Cached NVIDIA GR00T Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None. Headless physics trace.
- Robot embodiment: G1 WBC without articulated fingers
- Robot controller: Decoupled WBC and posture backend
- Hardware: Local Windows CPU
- Command: `python -m pytest -o addopts= -s tests/test_g1_posture.py::test_rollout_gate_then_monitored_real_execution_reaches_a_mid_height_target C:/Users/24adi/DREAM-r1-evidence/rollback-shifted-target/test_shifted.py::test_rollout_gate_then_monitored_real_execution_reaches_a_mid_height_target`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 2
- Cases completed: 2

## Purpose

Check the original reach regression and a separately specified shifted target and seed.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Both cases passed. Final independent hand errors with measured capabilities were 4.0 and 4.1 mm. The test-only wider-capability runs were reset separately and also passed their checks.

## Regressions observed

No regression was detected in the two tested development cases.

## Interruptions and failures

The process completed successfully after the conversation interruption.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/rollback-shifted-target (dir, sha256 a0c36367a268d8041f4e80de2903581de6c7725ae367f5cbd2c073f63c5b9036)
