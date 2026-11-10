# Run report: Test bounded Cartesian hand feedback at fixed body posture

- Run ID: 2026-11-10-006/20261110-115821-cartesian-feedback
- Parent task: 2026-11-10-006
- Started: 2026-11-10T11:58:21+11:00
- Ended: 2026-11-10T11:59:37+11:00
- Git commit: 6299f156b07785d65615fba43a642e1c6a2be237 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_feedback.json (f7df70fa799eb8056d956185907af48ab74f07d370caa9f3cc8ec19545ee4591)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC Balance and Walk
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC balance with measured-arm Cartesian correction
- Hardware: Local Windows CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_feedback.json --out C:/Users/24adi/DREAM-r1-evidence/r1-feedback-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Test whether measured Cartesian feedback can raise the arm behind the table without changing body posture.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The raise did not converge. The existing supervisor stopped on the right elbow position margin at 1.72 s, with 228 mm of position error still present.

## Regressions observed

This was an opt-in development controller. It did not replace the validated posture backend.

## Interruptions and failures

Stopped by the joint-position guard. The unsuccessful feedback source and config are preserved with the trace.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-feedback-001 (dir, sha256 0d32dcdb25392e18e2a8253f29360e971649d4e82389a5b094f7d1a3b08040d9)
