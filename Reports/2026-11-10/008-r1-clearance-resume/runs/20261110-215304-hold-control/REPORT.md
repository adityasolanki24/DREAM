# Run report: Check near-edge clearance with no arm command change

- Run ID: 2026-11-10-008/20261110-215304-hold-control
- Parent task: 2026-11-10-008
- Started: 2026-11-10T21:53:04+11:00
- Ended: 2026-11-10T21:54:09+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: Local CPU
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-hold-control-001/probe.py --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-hold-control-001/episode`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Separate motion caused by ongoing balance settling from motion caused by the arm trajectory.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The unchanged-command preview held without a stop for 0.62 s. Its base advanced about 39 mm in x, and the hand stayed at least 0.10 m from the table.

## Regressions observed

Not measured. This is a control without a new arm command.

## Interruptions and failures

The probe intentionally exits after the hold and does not write a full episode summary.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-hold-control-001 (dir, sha256 8e4b49eed51771427f053a995ff7c77b25d59e3d3b733bd85f7fa89bda38c21c)
