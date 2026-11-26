# Run report: Run the complete CPU test suite on the corrected humanoid tree

- Run ID: 2026-11-27-001/20261127-033044-full-regression
- Parent task: 2026-11-27-001
- Started: 2026-11-27T03:30:44+11:00
- Ended: 2026-11-27T03:47:55+11:00
- Git commit: cf2851db6f1b266e1fdf63aa76b3bf45c94f8a53 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: No external config file. Fixed test source and the saved collection order identify the cases.
- Seeds: Fixed seeds in the tests at the source snapshot
- Model and checkpoint: Repository models and cached NVIDIA WBC ONNX
- Simulator and version: MuJoCo 3.16.0
- Renderer: Native OpenGL where required by tests
- Robot embodiment: Panda, gantry, supported and free-standing G1
- Robot controller: Repository controllers under their test contracts
- Hardware: Local CPU, one pytest worker
- Command: `python -m pytest --junitxml=C:/Users/24adi/DREAM-r1-evidence/r1-validation-20261127/full.xml`
- Exit status: -1
- Completed (yes/no/partial): no (interrupted)
- Cases attempted: Not recorded
- Cases completed: Not recorded

## Purpose

Run the complete CPU test suite on the corrected humanoid source.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The full run reached at least 216 reported test outcomes and showed two failures. It was stopped before completion and has no complete-suite pass claim. The preceding fast run passed 1006 tests with two skipped and 89 deselected.

## Regressions observed

Failure positions map to the shift-in contact search and unrotated walking grasp tests. The physical outcome and the exact failed assertions are being reproduced separately.

## Interruptions and failures

Deliberately interrupted after the failures to fix the affected behaviour before further evaluation. The exact matching pytest process family was terminated and its handle returned -1. Logs and collection order are preserved. A new full run is required on the corrected final tree.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-validation-20261127 (dir, sha256 9892355c61c09ab29aec6692f0b920091ddf153eb94966da809b44e9b7c5adc0)
