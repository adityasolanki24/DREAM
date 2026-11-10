# Run report: Check explicit grasp depth on lateral

- Run ID: 2026-11-10-008/20261110-220812-depth-lateral
- Parent task: 2026-11-10-008
- Started: 2026-11-10T22:08:12+11:00
- Ended: 2026-11-10T22:08:31+11:00
- Git commit: 52d3c4aa200cd5f08b3201559015315024ae994b (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: experiments/configs/g1_grasp_bringup_lateral.json (375844721b083e09397a58b271f342269ee0b1099d561aabc689fa7086555caf)
- Seeds: 23
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: Free-standing G1 Dex3
- Robot controller: Frozen pose and tactile grasp controller
- Hardware: Local CPU
- Command: `python -m experiments.g1_grasp_bringup --config experiments/configs/g1_grasp_bringup_lateral.json --out C:/Users/24adi/DREAM-r1-evidence/r1-depth-lateral-001`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check the descent with an explicit depth acceptance window.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
Lifted to 143.4 mm, then lost grip during hold. The prior controller passed this case.

## Regressions observed

The hold result is reported for this development case.

## Interruptions and failures

See the grip and hold outcome in Results.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-depth-lateral-001 (dir, sha256 31553b11b51e04693d147804f155a38a79e39ae9fd1e1c52a9b94fbe2bed6649)
