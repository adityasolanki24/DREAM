# Run report: Fixed-stance reach diagnostic with 8 cm less forward reach

- Run ID: 2026-11-10-003/20261110-102409-closer-stance
- Parent task: 2026-11-10-003
- Started: 2026-11-10T10:24:09+11:00
- Ended: 2026-11-10T10:24:53+11:00
- Git commit: efd6872f4c375f948244e04c5beb71eade626925 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_closer.json (54047bfd4413682a069d3a3a0fe819d1867124b3477400318ab5a950f77e06a0)
- Seeds: 0
- Model and checkpoint: Cached NVIDIA GR00T Decoupled WBC Balance and Walk
- Simulator and version: MuJoCo 3.16.0
- Renderer: None; instrumented physics trace
- Robot embodiment: Free-standing Unitree G1 with Dex3
- Robot controller: NVIDIA WBC with monitored posture targets
- Hardware: Local Windows CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_closer.json --out C:/Users/24adi/DREAM-r1-evidence/closer-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test whether reducing forward reach by 8 cm avoids the nominal posture rejection. Keep box, table, policy, seed and safety limits fixed.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The contact supervisor stopped settling at 0.10 s at 39.17 N of right-hand touch against the 35 N diagnostic limit. No reach was planned and no grasp was verified.

## Regressions observed

Not measured. This changed an initial condition in the diagnostic only.

## Interruptions and failures

A measured contact stop preempted the planned comparison of reach feasibility. The touch observation gives force and link, but not contacted-object identity.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/closer-001 (dir, sha256 193241afb7921ca04991b4e82f7f17564b052f3da4d6645dafa33be50be772a4)
