# Run report: Instrument the reproduced reach rollback failure

- Run ID: 2026-11-10-004/20261110-103624-rollback-diagnosis
- Parent task: 2026-11-10-004
- Started: 2026-11-10T10:36:24+11:00
- Ended: 2026-11-10T10:37:35+11:00
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
- Command: `python C:/Users/24adi/DREAM-r1-evidence/rollback-debug/probe.py`
- Exit status: 1
- Completed (yes/no/partial): no (failed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Inspect predicted hand and torso motion during the previously reproduced reach correction failure.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The IK predicted a 32.45 mrad increase in torso pitch for the correction, and the target changed from 1.23 to 33.68 mrad. Actual hand error rose from 24.24 to 30.5 mm after correction, and to 38.08 mm after replaying the original target.

## Regressions observed

Previously reproduced on the source branch. This execution added diagnostic tracing only.

## Interruptions and failures

The monitored reach completed. The test failed on its independent post-correction error assertion.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/rollback-debug (dir, sha256 71d233f609e4acf45ad382896342a23cfa997956e8bccbac77e7440c2d766047)
