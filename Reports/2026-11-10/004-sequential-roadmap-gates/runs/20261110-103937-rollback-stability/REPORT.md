# Run report: Validate measured hand settling before correction

- Run ID: 2026-11-10-004/20261110-103937-rollback-stability
- Parent task: 2026-11-10-004
- Started: 2026-11-10T10:39:37+11:00
- Ended: 2026-11-10T10:40:23+11:00
- Git commit: efd6872f4c375f948244e04c5beb71eade626925 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: Not applicable. This run uses the repository physics regression fixture or the captured probe script.
- Seeds: 0, 1
- Model and checkpoint: Cached NVIDIA GR00T Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None. Headless physics trace.
- Robot embodiment: G1 WBC without articulated fingers
- Robot controller: Decoupled WBC and posture backend
- Hardware: Local Windows CPU
- Command: `python -m pytest -s tests/test_g1_posture.py::test_rollout_gate_then_monitored_real_execution_reaches_a_mid_height_target`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test whether observed hand stability before evaluating the correction solves the original physical regression.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The physical regression passed. Estimated hand error after the monitored hold decreased from 38.4 to 6.6 mm in three correction steps. Independent final hand error was 4.0 mm, with no protective stop.

## Regressions observed

The original test passed. Independent variation and suite checks follow separately.

## Interruptions and failures

None.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/rollback-stability-gate (dir, sha256 3c2bb1a10ce1c594c7992e9acad2726a070aaca028b55809e0a32f555d3d123e)
