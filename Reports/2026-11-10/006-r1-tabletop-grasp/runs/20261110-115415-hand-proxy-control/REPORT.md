# Run report: Isolate the inflated fixed-hand collision proxy in posture planning

- Run ID: 2026-11-10-006/20261110-115415-hand-proxy-control
- Parent task: 2026-11-10-006
- Started: 2026-11-10T11:54:15+11:00
- Ended: 2026-11-10T11:55:00+11:00
- Git commit: 6299f156b07785d65615fba43a642e1c6a2be237 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: Probe source is captured and hashed in Evidence.
- Seeds: Not applicable
- Model and checkpoint: Robot kinematics only
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: G1 planning model
- Robot controller: Mink posture IK, no motion
- Hardware: Local Windows CPU
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-ik-margin-001/probe.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 2
- Cases completed: 2

## Purpose

Compare the same IK input with and without the inflated fixed-finger hand proxy. Keep the body collision limits in both conditions.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Hand target residual was 22.6 mm with the inflated proxy and 3.2 mm without it. The latter predicted near-horizontal wrist orientation. No robot motion was made.

## Regressions observed

Not measured. This was a planning-only control.

## Interruptions and failures

The conservative hand approximation made the first candidate infeasible.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-ik-margin-001 (dir, sha256 b15224ca1f53272230a1203e60d9835294a8ad5d7f558286cce69dc98f5479cc)
