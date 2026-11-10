# Run report: Retain the WBC reset legs with the ready arm reference

- Run ID: 2026-11-10-006/20261110-115717-ready-native-legs
- Parent task: 2026-11-10-006
- Started: 2026-11-10T11:57:17+11:00
- Ended: 2026-11-10T11:58:17+11:00
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
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_ready.json --out C:/Users/24adi/DREAM-r1-evidence/r1-ready-002`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Keep the native WBC leg and root reset while using the ready arm target.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Reset cleared the table by 90 mm. It avoided the initial tilt-rate stop, but the settling motion triggered the hand clearance guard at 0.18 s.

## Regressions observed

Not measured. The ready reset remains an explicit development control.

## Interruptions and failures

No closure or lift. The static ready clearance did not persist during settle.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-ready-002 (dir, sha256 90398b6a8f42c0e0a1436c9e4048b18a08770293bb77435270a659f62e4dc87b)
