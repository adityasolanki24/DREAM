# Run report: Reproduce tabletop contact and measure the wrist frame

- Run ID: 2026-11-10-006/20261110-115046-pose-baseline
- Parent task: 2026-11-10-006
- Started: 2026-11-10T11:50:46+11:00
- Ended: 2026-11-10T11:51:52+11:00
- Git commit: 6299f156b07785d65615fba43a642e1c6a2be237 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_both_arms_clear.json (3008accc67d30b784f3866e66416453ad77714f5d15ff357eae85f0f4da57bc0)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC Balance and Walk
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC and posture supervisor
- Hardware: Local Windows CPU
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-pose-baseline-001/probe.py`
- Exit status: 1
- Completed (yes/no/partial): no (failed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Reproduce the tabletop false positive and compare the reach tip, wrist orientation, and Dex3 grasp frame.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The regression failed. The tool point used by reach ended at z=0.635 m and the prior commissioning grasp center at x=0.337 m, compared with the box at x=0.47 m and z=0.775 m.

## Regressions observed

Known failure reproduced.

## Interruptions and failures

Tabletop touch triggered a lift request without lifting the box.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-pose-baseline-001 (dir, sha256 b80c7de994cc82c1bc35b7739a1388bd9b98149c1df5b896b55e421c8632b49f)
