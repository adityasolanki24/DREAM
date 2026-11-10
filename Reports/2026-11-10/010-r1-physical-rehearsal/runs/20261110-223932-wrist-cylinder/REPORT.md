# Run report: R1 fixed loaded wrist orientation on cylinder

- Run ID: 2026-11-10-010/20261110-223932-wrist-cylinder
- Parent task: 2026-11-10-010
- Started: 2026-11-10T22:39:32+11:00
- Ended: 2026-11-10T22:39:46+11:00
- Git commit: 6fc17c5a0d6a718d032a54cd08b89f29a3688ea4 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/cylinder.json (c01d907e5d242a95cba3123218587264c4106c9d616225c62b5c16d78ae6be61)
- Seeds: 41
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/cylinder.json --out C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/cylinder`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Check whether preserving the initial loaded wrist orientation improves the static-ranking execution.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The independent live hold failed. Peak lift was 0.0516 m.

## Regressions observed

The live outcome matched the previous static ranking.

## Interruptions and failures

The live controller stopped for grip lost.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-loaded-wrist-001/cylinder (dir, sha256 4a38f92a1fbe63300de661ae10342ce44d6b37b83c6943680c0b1af23a94504f)
