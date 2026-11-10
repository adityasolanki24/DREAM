# Run report: Test a shorter finger envelope during the arm raise

- Run ID: 2026-11-10-006/20261110-120235-compact-raise
- Parent task: 2026-11-10-006
- Started: 2026-11-10T12:02:35+11:00
- Ended: 2026-11-10T12:03:22+11:00
- Git commit: 6299f156b07785d65615fba43a642e1c6a2be237 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_compact.json (66a9744f2c4a6da9a7799d734cee98a233ed4926db0c3118de13e95f4b4e4921)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC Balance and Walk
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC and posture supervisor
- Hardware: Local Windows CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_compact.json --out C:/Users/24adi/DREAM-r1-evidence/r1-compact-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test whether curling the fingers reduces the reach envelope enough to clear the table edge.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The guard still stopped the raise at 2.92 s, with 1.30 mm estimated separation. The shorter hand envelope did not fix the joint trajectory.

## Regressions observed

Not measured. This was an isolated control. Its source and configuration are preserved with the trace.

## Interruptions and failures

Stopped before opening or closing near the box.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-compact-001 (dir, sha256 fbe76d18e9b6d7f1ba6172d14f0b6afa928db38618c649cb468e166dcf365b73)
