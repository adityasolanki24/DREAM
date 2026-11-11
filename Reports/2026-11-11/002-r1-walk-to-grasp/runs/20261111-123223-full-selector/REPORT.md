# Run report: Test closed-loop walking and grasp selection with object-motion monitoring

- Run ID: 2026-11-11-002/20261111-123223-full-selector
- Parent task: 2026-11-11-002
- Started: 2026-11-11T12:32:23+11:00
- Ended: 2026-11-11T12:35:12+11:00
- Git commit: c866699583059bd16b3f3364b04ffc3cd0fb2dd6 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 81,83
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-walk-full-selector-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 2
- Cases completed: 2

## Purpose

Use geometry and full-body grasp forecasts with an unintended object-displacement check during the approach.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Both cases walked to the selected stance, lifted and held for three seconds. The measured displacements were about 1.09 m and 1.18 m. Neither walking phase recorded table contact.

## Regressions observed

This run compared walking starts only. Fixed-stance behaviour was checked separately.

## Interruptions and failures

Both previously failed starts passed. This comparison remains a two-case development result.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-walk-full-selector-001 (dir, sha256 afeea4f9a708f5afe59002713a934de1464ec7ef8dd06ea57bdfcd9ea5de76b2)
