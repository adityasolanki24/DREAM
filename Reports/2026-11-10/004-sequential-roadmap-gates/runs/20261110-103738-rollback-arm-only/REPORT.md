# Run report: Evaluate fixed torso correction against the original physics regression

- Run ID: 2026-11-10-004/20261110-103738-rollback-arm-only
- Parent task: 2026-11-10-004
- Started: 2026-11-10T10:37:38+11:00
- Ended: 2026-11-10T10:38:14+11:00
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

Test whether holding the torso command fixed eliminates the observed reach correction degradation.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Torso was held at the original command. Error still rose from 24.24 to 30.0 mm and reached 36.68 mm after rollback. Torso command change alone does not explain the failure.

## Regressions observed

Original physical assertion still fails; the proposed fix did not resolve the failure.

## Interruptions and failures

Test failed at the same independent post-correction error assertion.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/rollback-arm-only (dir, sha256 4c289cc356a21216c95a66bc908777b243c5bbb86793524df9b9ab2d9d29ebde)
