# Run report: Replay the grasp search using approach-frame depth groups

- Run ID: 2026-11-11-002/20261111-125608-approach-depth
- Parent task: 2026-11-11-002
- Started: 2026-11-11T12:56:08+11:00
- Ended: 2026-11-11T12:59:13+11:00
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
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-walk-approach-frame-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 4
- Cases completed: 4

## Purpose

Verify the same physical gate after grouping candidate depths along the selected approach direction.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Three of four walks ended with a verified hold. The rotated and unrotated offset cases selected the same grasp rank. All four reached their stances with zero table-contact samples.

## Regressions observed

Both previously passing walking controls were retained. Sensed and independent outcomes agreed on all four cases.

## Interruptions and failures

All four episodes completed. The cylinder lost opposing contact after lifting.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-walk-approach-frame-001 (dir, sha256 cede0cbada4156d33451978077c35fabbb98426927be76e6d04cb8d7dceafd7c)
