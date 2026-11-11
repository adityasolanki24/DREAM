# Run report: Check observed object-contact loss against all earlier holds and the sphere challenge

- Run ID: 2026-11-11-001/20261111-115152-contact-regression
- Parent task: 2026-11-11-001
- Started: 2026-11-11T11:51:52+11:00
- Ended: 2026-11-11T11:59:35+11:00
- Git commit: d80aa8d5cd282ebb51e50407b9fcb074d2010a61 (dirty)
- Branch: codex/g1-integration
- Config files and SHA-256: results/g1_r1_depth_dev_20261110/summary.json (68b8425b9e245a420f88e6a23fe3a3aaa9a889603c98d5bb94431c9f586de829); C:/Users/24adi/DREAM-r1-evidence/r1-compound-live-001/mug.json (10a40933002fa08e55aff43aa3fc856fb2c099ab37132e37a7d1192267acc5c2); C:/Users/24adi/DREAM-r1-evidence/r1-compound-live-001/mug_shift.json (ee32a6c44c80b82f320af345b4a47ea228dc128410abade6264cd1e9103bc650)
- Seeds: same matched, shifted and mug fixtures
- Model and checkpoint: Cached Decoupled WBC
- Simulator and version: MuJoCo 3.16.0
- Renderer: None
- Robot embodiment: free-standing G1 Dex3
- Robot controller: Decoupled WBC
- Hardware: local CPU, two workers
- Command: `python C:/Users/24adi/DREAM-r1-evidence/r1-observed-contact-regression-001/run.py`
- Exit status: 0
- Completed (yes/no/partial): yes (completed)
- Cases attempted: 13
- Cases completed: 13

## Purpose

Check whether the inferred object-contact stop disrupts previously verified grasps.

## Results

<!-- Passed and failed cases; quantitative metrics. -->
All twelve earlier positive controls held and their sensed results agreed with evaluation. The sphere failed as before.

## Regressions observed

No lost positive control was observed in this thirteen-case replay.

## Interruptions and failures

All thirteen episodes completed.

## Evidence

- C:/Users/24adi/DREAM-r1-evidence/r1-observed-contact-regression-001 (dir, sha256 4a76ca19c6d82a805254f07b24bc55392faf4b4e9dccced923cce51d229fc39b)
