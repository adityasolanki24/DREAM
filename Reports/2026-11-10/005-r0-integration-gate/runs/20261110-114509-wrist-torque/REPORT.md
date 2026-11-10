# Run report: Reproduce the tucked R1 wrist torque stop with actuator evidence

- Run ID: 2026-11-10-005/20261110-114509-wrist-torque
- Parent task: 2026-11-10-005
- Started: 2026-11-10T11:45:09+11:00
- Ended: 2026-11-10T11:45:55+11:00
- Git commit: 33e23c0409bcb5af502ce9c12a87aa277b940e6c (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_tucked.json (e1e74b7cc4aa69f0a8afaf771fcc1aa6f2a8a335163fb6bcfea7cd1d734b0e9a)
- Seeds: 0
- Model and checkpoint: Cached NVIDIA GR00T Decoupled WBC Balance and Walk
- Simulator and version: MuJoCo 3.16.0
- Renderer: None. Headless physics trace.
- Robot embodiment: Free-standing Unitree G1 with Dex3
- Robot controller: Decoupled WBC and posture supervisor
- Hardware: Local Windows CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_tucked.json --out C:/Users/24adi/DREAM-r1-evidence/wrist-torque-attribution-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Separate wrist PD demand and applied torque during the known tucked-arm stop.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The right wrist pitch requested 9.32 Nm in magnitude at the stop and MuJoCo capped applied torque at 5.00 Nm. The right hand contacted the right hip throughout the final approach to the stop.

## Regressions observed

Trace instrumentation reproduced the same stop at 1.60 s and requested-torque ratio 1.865.

## Interruptions and failures

The supervisor stopped the reach before finger closure.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/wrist-torque-attribution-001 (dir, sha256 e46749270179fd6bb85412487b64321e598e19738da993ab1cdc6c1822b61a4b)
