# Run report: Isolate hip contact by changing the starting right shoulder roll

- Run ID: 2026-11-10-005/20261110-114557-hip-clearance
- Parent task: 2026-11-10-005
- Started: 2026-11-10T11:45:57+11:00
- Ended: 2026-11-10T11:46:42+11:00
- Git commit: 33e23c0409bcb5af502ce9c12a87aa277b940e6c (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_hip_clearance.json (0d5b0092ae541ea30ac1f42cceaaa29199ab1a575d2a2e0017c4f35614248299)
- Seeds: 0
- Model and checkpoint: Cached NVIDIA GR00T Decoupled WBC Balance and Walk
- Simulator and version: MuJoCo 3.16.0
- Renderer: None. Headless physics trace.
- Robot embodiment: Free-standing Unitree G1 with Dex3
- Robot controller: Decoupled WBC and posture supervisor
- Hardware: Local Windows CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_hip_clearance.json --out C:/Users/24adi/DREAM-r1-evidence/hip-clearance-control-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test whether changing only the initial right shoulder roll removes hip contact and the right wrist torque stop.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The right hand never contacted the right hip, and the right wrist did not stop the run. The left wrist pitch triggered the supervisor at 1.98 s and 1.593 times its limit. No closure or lift.

## Regressions observed

The right wrist stop cleared. A different joint stopped the reach; the fixture remains a development control.

## Interruptions and failures

Left wrist torque stop. This trace recorded contacted geometry for the right hand only.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/hip-clearance-control-001 (dir, sha256 60ff2331d3df251a45a3c4515e99eb426a32a3346428b6b2b258adee65531f07)
