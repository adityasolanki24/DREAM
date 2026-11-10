# Run report: R1 fixed loaded wrist orientation on shift_out

- Run ID: 2026-11-10-010/20261110-223806-wrist-shift-out
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:38:06+11:00
- Ended: 2026-11-10T22:38:18+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/shift_out.json (3299e21fcbf06b4cf316353a6cc58e6aeb087021f4f23f23539d055d3222d15c)
- Seeds: 17
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/shift_out.json --out C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/shift_out`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check whether preserving the initial loaded wrist orientation improves the static-ranking execution.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold failed. Peak lift was 0.0022 m.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The live controller stopped for grip lost.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/shift_out (dir, sha256 264f324d5aa772962c9193c08c40fa7c2608922e1ab864b31452b41cfc1e949c)
