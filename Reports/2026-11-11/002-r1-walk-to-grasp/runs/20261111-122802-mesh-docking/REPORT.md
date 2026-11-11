# Run report: Check the reachable approach with measured robot mesh clearance

- Run ID: 2026-11-11-002/20261111-122802-mesh-docking
- Parent task: 2026-11-11-002
- Started: 2026-11-11T12:28:02+11:00
- Ended: 2026-11-11T12:29:15+11:00
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
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-walk-clearance-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 2
- Cases completed: 2

## Purpose

Check arm reach at the grasp and pregrasp while supervising robot mesh clearance close to the table.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Both starts arrived closer to the object and recorded zero robot-table contact samples during the walk and settle. Both rejected the raise.

## Regressions observed

This run compared walking starts only. Fixed-stance behaviour was checked separately.

## Interruptions and failures

The geometrically safe stance did not align the working shoulder well enough for the full motion.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-walk-clearance-001 (dir, sha256 52195b77c50f25d84ba80dd7e130be5af3b8fd4a3e39ed7e03cedfffe64ea61a)
