# Run report: Test the Dex3 grasp frame and closure guards

- Run ID: 2026-11-10-006/20261110-115238-grasp-frame
- Parent task: 2026-11-10-006
- Started: 2026-11-10T11:52:38+11:00
- Ended: 2026-11-10T11:53:26+11:00
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
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_both_arms_clear.json --out C:/Users/24adi/DREAM-r1-evidence/r1-grasp-frame-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Evaluate a wrist orientation and grasp offset based on the articulated hand.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
After settling the estimated stance was x=-0.007 m, from a reset at x=0.08 m. The horizontal pinch target had 48.2 mm of planned residual, and the planner refused.

## Regressions observed

The prior fixture is preserved. The stricter pose request was refused before contact.

## Interruptions and failures

No reach, closure or lift on this fixture.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-grasp-frame-001 (dir, sha256 d5a150acfc943766c064aa3219e007a7fee256b85ce6e54a62884ea4dda496ad)
