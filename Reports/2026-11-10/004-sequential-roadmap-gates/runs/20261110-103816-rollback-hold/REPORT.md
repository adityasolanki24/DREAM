# Run report: Compare drift under repeated unchanged reach commands

- Run ID: 2026-11-10-004/20261110-103816-rollback-hold
- Parent task: 2026-11-10-004
- Started: 2026-11-10T10:38:16+11:00
- Ended: 2026-11-10T10:39:34+11:00
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
- Command: `python C:/Users/24adi/DREAM-r1-evidence/rollback-hold-control/probe.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Isolate continued robot motion from the correction command by holding the same posture and observing both encoder-based estimates and evaluator truth.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
With unchanged command, true base x moved from -28.37 mm to -54.48 mm on the first hold, then settled near -45 mm. True hand x residual changed by 28.7 mm on the first hold. The sensor-based estimate showed the same movement. After settling the arm-only correction reduced true hand error to 16.4 mm.

## Regressions observed

Not measured. The hold was an instrumented control used to isolate settling, not an acceptance protocol.

## Interruptions and failures

No protective stop during the observed control. The open-loop pose continued to move after the nominal settling time.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/rollback-hold-control (dir, sha256 c6ec948e79e544599c09586347527c2ef0936e37134cf6199cc4f5c928656022)
