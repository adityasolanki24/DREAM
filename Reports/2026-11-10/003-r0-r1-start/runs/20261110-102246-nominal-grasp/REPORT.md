# Run report: Nominal fixed-stance Dex3 grasp diagnostic

- Run ID: 2026-11-10-003/20261110-102246-nominal-grasp
- Parent task: 2026-11-10-003
- Started: 2026-11-10T10:22:46+11:00
- Ended: 2026-11-10T10:24:05+11:00
- Git commit: efd6872f4c375f948244e04c5beb71eade626925 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup.json (9e72e70fbb8e3797ccb87c551792440ee761ad43663bb64361a7a4a1c4a0c9d4)
- Seeds: 0
- Model and checkpoint: Cached NVIDIA GR00T Decoupled WBC Balance and Walk
- Simulator and version: MuJoCo 3.16.0
- Renderer: None; instrumented physics trace
- Robot embodiment: Free-standing Unitree G1 with Dex3
- Robot controller: NVIDIA WBC with monitored posture targets
- Hardware: Local Windows CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup.json --out C:/Users/24adi/DREAM-r1-evidence/nominal-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Isolate nominal fixed-stance reach and contact closure on the free-standing G1 before introducing approach or perception.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Seed 0 settled for 1 s without a stop. The sole posture was rejected for hand residual and COM outside support. No reach, closure or lift was executed. No verified grasp.

## Regressions observed

Not measured.

## Interruptions and failures

Execution ended normally at the planning refusal. Physical finger grasping remains untested by this episode.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/nominal-001 (dir, sha256 8c94d42aa7900411a266ab92a75845d700691c2e2da86d19a0f471052221921b)
