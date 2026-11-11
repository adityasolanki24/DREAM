# Run report: Verify walking grasp across matched, rotated and changed-object fixtures

- Run ID: 2026-11-11-002/20261111-123545-walking-transfer
- Parent task: 2026-11-11-002
- Started: 2026-11-11T12:35:45+11:00
- Ended: 2026-11-11T12:43:13+11:00
- Git commit: c866699583059bd16b3f3364b04ffc3cd0fb2dd6 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b); experiments/configs/g1_grasp_bringup_cylinder.json (ecb6de06dd040e4a769537eaa933a0462ccfe70d962cf308c713a20fb345dcd8)
- Seeds: 81,83,89,97
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-walk-transfer-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 4
- Cases completed: 4

## Purpose

Check the same starts with independent trajectory materials and test a rotated scene and a changed cylinder.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Both matched starts held. The rotated case selected a diagonal stance and refused the pregrasp. The changed cylinder walked about 1.32 m, lifted about 0.055 m and slipped.

## Regressions observed

The two matched holds were preserved. Neither changed fixture passed in this replay.

## Interruptions and failures

All four cases completed. Each walk reached its stance with zero robot-table contact samples.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-walk-transfer-001 (dir, sha256 2b77d6fd9956756646ea2abd8a723eca589a840db2ac801a940ec68140739980)
