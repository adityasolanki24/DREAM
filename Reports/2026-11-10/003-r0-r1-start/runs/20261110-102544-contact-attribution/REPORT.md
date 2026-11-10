# Run report: Repeat closer stance with independent collision attribution

- Run ID: 2026-11-10-003/20261110-102544-contact-attribution
- Parent task: 2026-11-10-003
- Started: 2026-11-10T10:25:44+11:00
- Ended: 2026-11-10T10:26:33+11:00
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
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_closer.json --out C:/Users/24adi/DREAM-r1-evidence/contact-attribution-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Identify the collider causing the settling force stop in the closer stance using evaluator-only contact identity.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The force stop reproduced at 0.10 s and 39.17 N. Both middle-finger links supplied the measured force and the box was the only right-hand contact partner at the stop. The target box interferes with the neutral arm and open hand during settling.

## Regressions observed

Not measured. Added evaluator logging only and reproduced the preceding stop.

## Interruptions and failures

The repeated contact stop preceded planning and closure.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/contact-attribution-001 (dir, sha256 067325a7e997e41f68e763d414b054db9723ac52b0f8a24b69d4f553f1d9f530)
