# Run report: Detector comparison pipeline check, CPU, truth-blind stub detector, 39 l_shape development frames

- Run ID: 2026-11-09-001/20261109-192742-detector-compare-stub-cpu
- Parent task: 2026-11-09-001
- Started: 2026-11-09T19:27:42+11:00
- Ended: 2026-11-09T19:27:54+11:00
- Git commit: 87d746ec84057d8675b060ea6b718907dc44cef1 (dirty)
- Branch: worktree-agent-ab8c50d8ef4f5ed08
- Config files and SHA-256: experiments/configs/detector_compare.json (7d68d4d912abbaf8a2297d5daca467c3ff14ee5ac08a09fc052bdef6dec3774e)
- Seeds: none
- Model and checkpoint: none (fixed stub boxes; not a detector)
- Simulator and version: none (offline)
- Renderer: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Robot embodiment: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Robot controller: Not applicable (offline replay of recorded frames; no simulator, robot or controller ran)
- Hardware: local CPU
- Command: `python experiments/detector_compare.py --stub --detectors owlv2-base --out results/detector_compare_stub_dev`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 39
- Cases completed: 39

## Purpose

Pipeline check of experiments/detector_compare.py end to end on the 39 development frames with a truth-blind stub detector (no model), before any GPU job.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Pipeline completed. The stub scores P/R 0.03/0.02 and produces 39 absent-class false positives for 'bed', as expected of a stub; these numbers say nothing about any detector.

## Regressions observed

Not applicable (pipeline check).

## Interruptions and failures

None.

## Evidence

- results/detector_compare_stub_dev/detector_compare.json (file, sha256 465c4d0bd1f6202df82c04bb200cb969200aff4192f7d4f20da804ac87b3e6d4)
