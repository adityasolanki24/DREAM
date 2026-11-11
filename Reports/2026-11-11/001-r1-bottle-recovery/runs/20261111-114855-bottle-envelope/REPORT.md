# Run report: Check bottle grasps against declared payload and friction bounds

- Run ID: 2026-11-11-001/20261111-114855-bottle-envelope
- Parent task: 2026-11-11-001
- Started: 2026-11-11T11:48:55+11:00
- Ended: 2026-11-11T11:54:02+11:00
- Git commit: d80aa8d5cd282ebb51e50407b9fcb074d2010a61 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001/bottle.json (e9e3e2e3a528fb665aa007f2b37d78488e6b5e2e64474d88dfc3ad0dbeb3c586); C:/Users/24adi/DREAM-r1-evidence/r1-compound-clearance-001/bottle_shift.json (32a8ae767f81d5ee3490accff761aa49b010822be2f73fdf76093d8edd7b4ec5)
- Seeds: 53,61 with an independent planning stream
- Model and checkpoint: Cached Decoupled WBC, prior mass 0.30 kg and friction 0.5
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU, serial
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-bottle-envelope-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 2
- Cases completed: 2

## Purpose

Use declared prior mass 0.30 kg and sliding friction 0.5 to test conservative prediction over the intended envelope.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
The proposed candidates failed the conservative prediction. Both live fallbacks failed.

## Regressions observed

No live retention gain was measured from this conservative prior.

## Interruptions and failures

Both episodes completed. The model bounds are explicit assumptions rather than values copied from actual fixture material.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-bottle-envelope-001 (dir, sha256 a539f1680dbe54c06f6721bdf3d48996dec924bf03252d01d5cda13b04778adb)
