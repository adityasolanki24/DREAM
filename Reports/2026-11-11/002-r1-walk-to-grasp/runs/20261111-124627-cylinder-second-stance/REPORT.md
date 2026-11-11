# Run report: Check the cylinder from the second reachable stance

- Run ID: 2026-11-11-002/20261111-124627-cylinder-second-stance
- Parent task: 2026-11-11-002
- Started: 2026-11-11T12:46:27+11:00
- Ended: 2026-11-11T12:51:09+11:00
- Git commit: c866699583059bd16b3f3364b04ffc3cd0fb2dd6 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-walk-cylinder-alternate-001/cylinder.json (8851df3462c03d0159029c0dc5d8cf16848d98a99c5ec581fcb65d8ffab6ee82)
- Seeds: 97
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-walk-cylinder-alternate-001/run.py --config C:/Users/24adi/DREAM-r1-evidence/r1-walk-cylinder-alternate-001/cylinder.json --out C:/Users/24adi/DREAM-r1-evidence/r1-walk-cylinder-alternate-001/run`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Replay the cylinder from the next stance accepted by the geometric and arm reach checks.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The alternate stance arrived after about 1.30 m of physical displacement with zero table-contact samples. The cylinder lifted about 0.054 m and slipped.

## Regressions observed

The alternate stance did not recover the hold. The result is a diagnostic replay with a forced candidate ordinal.

## Interruptions and failures

The case completed and the grip-loss guard stopped execution.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-walk-cylinder-alternate-001 (dir, sha256 34847c015b66d57a03c2156ffca48bee781a41b3b98167ee29b2025942909dc0)
