# Run report: Attribute the failed predicted bottle holds

- Run ID: 2026-11-11-001/20261111-113621-predicted-bottle
- Parent task: 2026-11-11-001
- Started: 2026-11-11T11:36:21+11:00
- Ended: 2026-11-11T11:38:10+11:00
- Git commit: d80aa8d5cd282ebb51e50407b9fcb074d2010a61 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001/bottle.json (e9e3e2e3a528fb665aa007f2b37d78488e6b5e2e64474d88dfc3ad0dbeb3c586); C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001/bottle_shift.json (32a8ae767f81d5ee3490accff761aa49b010822be2f73fdf76093d8edd7b4ec5)
- Seeds: independent planning stream with entropy 0 and spawn key 1
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-bottle-prediction-audit-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 3
- Cases completed: 3

## Purpose

Distinguish slip, unsupported clearance and reach limits in the recorded bottle candidate failures.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The first bottle candidate completed 150 hold samples but dropped to the table. Its minimum hold rise was negative 3.4 mm and minimum bottom clearance negative 6.0 mm. Thumb and opposing finger pressure persisted at about 12 N after the drop. Another candidate slipped during lift. The shifted bottle lost its grasp after 134 hold samples.

## Regressions observed

The independent evaluator and sensed success claim rejected these holds. The real-time controller did not recognize the first drop from tactile pressure alone.

## Interruptions and failures

All three audit episodes completed. These are failed retention predictions and are preserved as reproducible development cases.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-bottle-prediction-audit-001 (dir, sha256 62ed4f8d499d96573b1e68d75b55a2d840078a6d92c302b493adb3523e094b30)
