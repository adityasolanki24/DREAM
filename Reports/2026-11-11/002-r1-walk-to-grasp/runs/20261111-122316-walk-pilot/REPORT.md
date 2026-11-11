# Run report: Test reachability-based stance selection and walking from two starts

- Run ID: 2026-11-11-002/20261111-122316-walk-pilot
- Parent task: 2026-11-11-002
- Started: 2026-11-11T12:23:16+11:00
- Ended: 2026-11-11T12:24:21+11:00
- Git commit: c866699583059bd16b3f3364b04ffc3cd0fb2dd6 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 81,83
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC, odometry follower
- Hardware: local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-walk-pilot-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 2
- Cases completed: 2

## Purpose

Check whether the existing forward follower can reach a grasp stance selected by arm IK.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The front start walked about 1.06 m before the footprint guard stopped it at the table. The offset start arrived but did not hold the object.

## Regressions observed

This run compared walking starts only. Fixed-stance behaviour was checked separately.

## Interruptions and failures

Neither episode verified a hold. The front approach drifted laterally during the final heading adjustment.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-walk-pilot-001 (dir, sha256 891baaf1911906fb07760471d9254b95e84745a6bb7eed6e3f9e17c0d493e0e6)
