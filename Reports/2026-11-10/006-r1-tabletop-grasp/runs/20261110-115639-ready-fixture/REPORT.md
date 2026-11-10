# Run report: Isolate closure from the arm raise in a planned ready fixture

- Run ID: 2026-11-10-006/20261110-115639-ready-fixture
- Parent task: 2026-11-10-006
- Started: 2026-11-10T11:56:39+11:00
- Ended: 2026-11-10T11:57:14+11:00
- Git commit: 6299f156b07785d65615fba43a642e1c6a2be237 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_ready.json (274e93f44a816592d95536920c3bc773545eb591818bd200d9a4a3fc4b010860)
- Seeds: 0
- Model and checkpoint: Cached Decoupled WBC Balance and Walk
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: WBC and posture supervisor
- Hardware: Local Windows CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_ready.json --out C:/Users/24adi/DREAM-r1-evidence/r1-ready-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Isolate contact closure using a static IK ready pose as the initial fixture.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The initial IK pose cleared the table by 46 mm, but the balance policy triggered the pelvis tilt-rate guard on the first tick.

## Regressions observed

Static IK pose is not a dynamically valid initial state for the balance controller.

## Interruptions and failures

The simulation stopped during settle. No arm movement was commanded after reset.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-ready-001 (dir, sha256 1651e48e35f139045d35319ff03b49b9a0018d0e458388e3ffa098b909b8d1bc)
