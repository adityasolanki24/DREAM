# Run report: Identify the link responsible for the clearance stop

- Run ID: 2026-11-10-006/20261110-120137-clearance-link
- Parent task: 2026-11-10-006
- Started: 2026-11-10T12:01:37+11:00
- Ended: 2026-11-10T12:02:31+11:00
- Git commit: 6299f156b07785d65615fba43a642e1c6a2be237 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_pinch.json (6e6b98f06df2b283a93ed1a1ba708724c0238fcf20f1c7b83e37565200a5dc7b)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC Balance and Walk
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC and posture supervisor
- Hardware: Local Windows CPU
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-clearance-link-001/probe.py --config experiments/configs/g1_grasp_bringup_pinch.json --out C:/Users/24adi/DREAM-r1-evidence/r1-clearance-link-001/episode`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Identify which articulated hand link triggers the guard on the original raise trajectory.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The estimated distal right index link reached the near tabletop edge while raising. The evaluator recorded no physical tabletop contact on the stopped trajectory.

## Regressions observed

Evaluator instrumentation reproduced the existing guard stop.

## Interruptions and failures

Stopped during raise, before closing near the box.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-clearance-link-001 (dir, sha256 328af6d8c58cc0a9cde5f4ccda7fc01f7b79693d1e85b6e37a680a159a3e7a07)
