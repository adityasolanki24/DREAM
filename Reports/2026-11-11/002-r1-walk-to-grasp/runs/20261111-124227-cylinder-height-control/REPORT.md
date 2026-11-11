# Run report: Replay the shifted-height cylinder at the earlier fixed stance

- Run ID: 2026-11-11-002/20261111-124227-cylinder-height-control
- Parent task: 2026-11-11-002
- Started: 2026-11-11T12:42:27+11:00
- Ended: 2026-11-11T12:43:14+11:00
- Git commit: c866699583059bd16b3f3364b04ffc3cd0fb2dd6 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-walk-cylinder-height-control-001/cylinder.json (a421a81bb9ad61481f9ee8758715c4bcc80252b30284d64a085444ee926cf279)
- Seeds: 97
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU
- Command: `python experiments/g1_grasp_bringup.py --config C:/Users/24adi/DREAM-r1-evidence/r1-walk-cylinder-height-control-001/cylinder.json --out C:/Users/24adi/DREAM-r1-evidence/r1-walk-cylinder-height-control-001/run`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 1
- Cases completed: 1

## Purpose

Replay the changed cylinder and table height at the previously validated fixed stance.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The cylinder held. It reached about 0.152 m peak lift and passed both sensed and independent hold checks.

## Regressions observed

The result limits the failure attribution to the mobile approach or its handover conditions. The fixed and walking paths also differ in their trajectory material models and motion guards.

## Interruptions and failures

The control completed without a protective stop.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-walk-cylinder-height-control-001 (dir, sha256 f10b22bcc6fd3191be5e3f55809c92f44faa357767eb5c90f7d592d27dcb5a6d)
