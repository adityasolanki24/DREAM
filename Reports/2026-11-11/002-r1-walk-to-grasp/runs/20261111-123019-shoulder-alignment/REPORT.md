# Run report: Replay with arm-aligned stance selection and a held idle arm

- Run ID: 2026-11-11-002/20261111-123019-shoulder-alignment
- Parent task: 2026-11-11-002
- Started: 2026-11-11T12:30:19+11:00
- Ended: 2026-11-11T12:30:45+11:00
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
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-walk-shoulder-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 2
- Cases completed: 2

## Purpose

Align the object with the working shoulder and include an arm-motion candidate that keeps the idle arm at its measured command.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Both cases walked and reached with zero table contact during locomotion. The front grasp lost contact in the probe. The offset approach displaced the object and did not establish opposing closure.

## Regressions observed

This run compared walking starts only. Fixed-stance behaviour was checked separately.

## Interruptions and failures

Neither episode passed the physical hold. The successful reach exposed distinct contact and pre-closure failures.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-walk-shoulder-001 (dir, sha256 7a8ff37bb0ff9fa6bbfb745c8f5109a76a430fc0feedfbadcc141c855f723e65)
