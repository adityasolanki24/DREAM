# Run report: Test both arms starting clear of the hips

- Run ID: 2026-11-10-005/20261110-114645-both-arms-clear
- Parent task: 2026-11-10-005
- Started: 2026-11-10T11:46:45+11:00
- Ended: 2026-11-10T11:47:29+11:00
- Git commit: 33e23c0409bcb5af502ce9c12a87aa277b940e6c (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_both_arms_clear.json (3008accc67d30b784f3866e66416453ad77714f5d15ff357eae85f0f4da57bc0)
- Seeds: 0
- Model and checkpoint: Cached NVIDIA GR00T Decoupled WBC Balance and Walk
- Simulator and version: MuJoCo 3.16.0
- Renderer: None. Headless physics trace.
- Robot embodiment: Free-standing Unitree G1 with Dex3
- Robot controller: Decoupled WBC and posture supervisor
- Hardware: Local Windows CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_both_arms_clear.json --out C:/Users/24adi/DREAM-r1-evidence/both-arms-clear-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test whether moving the idle left arm clear of the hip allows monitored reach and contact closure while preserving the right arm clearance control.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The reach completed and opposed touch was detected. Independent contact identity shows that the hand contacted the tabletop only during closure and the lift attempt. The box never lifted. The lift attempt stopped at 36.13 N of summed right-hand touch against the 35 N cap.

## Regressions observed

Neither wrist triggered the stop in this fixture. Opposed touch alone falsely indicated potential purchase and caused a lift attempt. Independent verification still declined grasp success.

## Interruptions and failures

The contact force stop ended the trial before a lift. This is a reproducible table-clearance and contact-attribution challenge.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/both-arms-clear-001 (dir, sha256 5f946b1175b1b3bcf39331077e8d589fefb64f7efc8efc0718d57b903c2bd498)
